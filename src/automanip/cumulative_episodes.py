"""Small episode persistence adapter for the separately versioned reuse track.

No environment/evaluator imports. Each directory has an immutable index.json
contract and immutable episode JSON/NPZ pairs. Digests detect corruption; they
are not a security boundary against somebody who can rewrite the whole store.
Bounds are intentionally small: shard larger collections into separate stores.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import struct
import time
import uuid
import zipfile

import numpy as np

FORMAT = "automanip-cumulative-episodes-v1"
MAX_STEPS = 4096
MAX_EPISODES = 10000
MAX_FILE_BYTES = 4 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_JSON_BYTES = 1024 * 1024
MAX_BUFFERED_TRANSITIONS = 65536
PARTITIONS = {"training", "development", "final"}
ARRAYS = {"observations": (48,), "actions": (8,), "next_observations": (48,),
          "rewards": (), "terminated": (), "truncated": (), "success": (), "success_known": ()}
FLAGS = {"terminated", "truncated", "success", "success_known"}
LEGACY_COLLECTOR_SHA256 = "523459ce91a59200970d66a7dd83fa8dc06c7c96879e72ab541f2f263403ef3b"
MAX_LEGACY_COMPRESSED_BYTES = 32 * 1024 * 1024
MAX_LEGACY_EXPANDED_BYTES = 256 * 1024 * 1024
MAX_LEGACY_EPISODES = 4096


def _json_bytes(value):
    try:
        raw = json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
    except (TypeError, ValueError) as error:
        raise ValueError("Metadata must be finite JSON data") from error
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Metadata exceeds store bound")
    return raw


def _json_read(raw):
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Metadata exceeds store bound")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("Nonfinite JSON constant: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _behavior(value):
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError("Behavior provenance must be a JSON dictionary")
    raw = _json_bytes(value)
    if len(raw) > 16384:
        raise ValueError("Behavior provenance exceeds 16KiB")
    return _json_read(raw)


def _id(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Task/interface identity must be a nonempty string <=256 characters")
    return value


def _numpy(value):
    # Torch is optional for reading stores, and no GPU tensor is created here.
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _float(value, shape, name):
    array = _numpy(value)
    if array.shape != shape or array.dtype.kind not in "fiu" or not np.isfinite(array).all():
        raise ValueError("Invalid finite numeric " + name)
    with np.errstate(over="ignore"):
        array = array.astype(np.float32, copy=True)
    if not np.isfinite(array).all():
        raise ValueError("Float32 overflow in " + name)
    return array


def _flags(value, count, name):
    array = _numpy(value)
    if array.shape == (count, 1):
        array = array[:, 0]
    if array.shape != (count,) or array.dtype.kind not in "bifu" or not np.isfinite(array).all() or not np.isin(array, (0, 1)).all():
        raise ValueError("Expected binary " + name)
    return array.astype(bool, copy=True)


def _write(path, raw):
    with path.open("xb") as handle:
        handle.write(raw)


class EpisodeRecorder:
    """Persist accepted transitions without collecting or resetting environments.

    Inputs are batched state48/action8. ``start`` denotes a whole-batch reset;
    it preserves any nonempty old attempts as incomplete. Non-autoreset callers
    must call start after any lane ends. ``behavior`` overrides persist until
    the next override. Automatic-reset seeds are unknown, never fabricated.
    """
    def __init__(self, directory, task_id: str, interface_id: str, partition,
                 behavior: dict, num_envs: int, auto_reset: bool = True):
        if partition not in PARTITIONS or type(num_envs) is not int or not 1 <= num_envs <= 4096 or type(auto_reset) is not bool:
            raise ValueError("Invalid partition, environment count, or autoreset mode")
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("Episode directory cannot be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise FileExistsError("Recorder requires a new or empty immutable episode directory")
        self.behavior = _behavior(behavior)
        self.manifest = dict(format=FORMAT, task_id=_id(task_id), interface_id=_id(interface_id),
                             partition=partition, num_envs=num_envs, auto_reset=auto_reset,
                             behavior=self.behavior, session_id=uuid.uuid4().hex,
                             array_schema="float32 state48/action8; bool boundary/outcome flags")
        raw = _json_bytes(self.manifest)
        _write(self.directory / "index.json", raw)
        self.index_sha = hashlib.sha256(raw).hexdigest()
        self.num_envs, self.auto_reset = num_envs, auto_reset
        self.buffers = [None] * num_envs
        self.ordinals = [0] * num_envs
        self.expected = None
        self.closed = False
        self.saved_count, self.saved_bytes = 0, len(raw)

    def start(self, obs, seed):
        if self.closed:
            raise ValueError("Recorder is finished")
        observed = _float(obs, (self.num_envs, 48), "initial observations")
        vector_seed = isinstance(seed, (list, tuple, np.ndarray))
        seeds = list(seed) if vector_seed else [seed] * self.num_envs
        if len(seeds) != self.num_envs or any(s is not None and (not isinstance(s, (int, np.integer)) or isinstance(s, (bool, np.bool_)) or not 0 <= int(s) < 2**63) for s in seeds):
            raise ValueError("Reset seed must be an integer, None, or one seed per lane")
        for lane, buffer in enumerate(self.buffers):
            if buffer is not None and buffer["rows"]:
                self._persist(lane, "explicit_reset", False)
        for lane, value in enumerate(seeds):
            self._new(lane, None if value is None else int(value),
                      "explicit_lane_seed" if vector_seed else "batched_reset_seed_argument")
        self.expected = observed

    def _new(self, lane, seed=None, seed_kind="autoreset_unknown"):
        self.buffers[lane] = dict(rows=[], segments=[], seed=seed, seed_kind=seed_kind)

    def step(self, obs, executed_actions, following, rewards, terminated, truncated, infos, behavior=None):
        if self.closed or self.expected is None or any(b is None for b in self.buffers):
            raise ValueError("Call start before recording, and after a non-autoreset boundary")
        count = self.num_envs
        before = _float(obs, (count, 48), "observations")
        raw_action = _numpy(executed_actions)
        action = _float(raw_action, (count, 8), "executed actions")
        after = _float(following, (count, 48), "following observations")
        reward = _numpy(rewards)
        if reward.shape == (count, 1):
            reward = reward[:, 0]
        reward = _float(reward, (count,), "rewards")
        term, trunc = _flags(terminated, count, "terminated"), _flags(truncated, count, "truncated")
        if (np.abs(raw_action) > 1).any():
            raise ValueError("Actions must already be executed/clipped to [-1,1]")
        if not np.array_equal(before, self.expected):
            raise ValueError("Before-observation/lane continuity differs")
        if not isinstance(infos, dict):
            raise ValueError("infos must be a dictionary")
        done = term | trunc
        actual = after.copy()
        if self.auto_reset:
            # Reuse the existing reviewed pre-reset-successor implementation.
            import torch
            from research.discovery.amp_v1 import actual_successor
            boundary = {}
            if done.any():
                boundary["final_observation"] = torch.from_numpy(_float(infos.get("final_observation"), (count, 48), "final observations"))
                boundary["_final_observation"] = torch.from_numpy(_flags(infos.get("_final_observation"), count, "final observation mask"))
            actual = actual_successor(torch.from_numpy(after), torch.from_numpy(done), boundary).numpy().copy()
        success, known = np.zeros(count, dtype=bool), np.zeros(count, dtype=bool)
        if "success" in infos:
            success = _flags(infos["success"], count, "success")
            known[:] = True
        if self.auto_reset and done.any():
            final = infos.get("final_info", {})
            if not isinstance(final, dict):
                raise ValueError("final_info must be a dictionary")
            if "success" in final:
                final_success = _flags(final["success"], count, "final success")
                success[done], known[done] = final_success[done], True
            else:
                # Returned infos can describe reset states, not terminal outcomes.
                success[done], known[done] = False, False
        provenance = _behavior(behavior) if behavior is not None else self.behavior
        if sum(len(b["rows"]) for b in self.buffers) + count > MAX_BUFFERED_TRANSITIONS:
            raise ValueError("Buffered transition bound reached; finish the accepted partial attempts")
        reserved = self.saved_bytes
        if self.saved_count + count > MAX_EPISODES:
            raise ValueError("Episode count bound reached; finish accepted partial attempts")
        for buffer in self.buffers:
            if len(buffer["rows"]) >= MAX_STEPS:
                raise ValueError("Episode length bound reached; finish the accepted partial attempts")
            if not buffer["segments"] or buffer["segments"][-1]["behavior"] != provenance:
                if len(_json_bytes(buffer["segments"])) + len(_json_bytes(provenance)) > MAX_JSON_BYTES - 8192:
                    raise ValueError("Behavior history bound reached; finish partial attempts")
            # Reserve enough for accepted rows, NPY headers, provenance and an
            # arbitrary bounded finish reason, so a full store can still finish.
            reserved += 12288 + 424*(len(buffer["rows"])+1) + len(_json_bytes(buffer["segments"])) + len(_json_bytes(provenance))
        if reserved > MAX_TOTAL_BYTES:
            raise ValueError("Store byte bound reached; finish accepted partial attempts")
        self.behavior = provenance
        ended = []
        for lane, buffer in enumerate(self.buffers):
            if not buffer["segments"] or buffer["segments"][-1]["behavior"] != provenance:
                buffer["segments"].append(dict(start=len(buffer["rows"]), behavior=provenance))
            buffer["rows"].append((before[lane], action[lane], actual[lane], reward[lane], term[lane], trunc[lane], success[lane], known[lane]))
            if done[lane]:
                reason = "terminated_and_truncated" if term[lane] and trunc[lane] else "terminated" if term[lane] else "truncated"
                ended.append(self._persist(lane, reason, True))
                if self.auto_reset:
                    self._new(lane)
        self.expected = after
        return ended

    def _persist(self, lane, reason, complete):
        buffer = self.buffers[lane]
        rows = buffer["rows"]
        arrays = {name: np.asarray([row[i] for row in rows], dtype=bool if name in FLAGS else np.float32)
                  for i, name in enumerate(ARRAYS)}
        packed = io.BytesIO()
        np.savez(packed, **arrays)
        raw = packed.getvalue()
        if self.saved_count >= MAX_EPISODES or len(raw) > MAX_FILE_BYTES or self.saved_bytes + len(raw) > MAX_TOTAL_BYTES:
            raise ValueError("Episode store bound reached; use another bounded store")
        ordinal = self.ordinals[lane]
        stem = f"episode-{lane:06d}-{ordinal:08d}"
        outcome = "incomplete" if not complete else "unknown" if not arrays["success_known"][-1] else "success" if arrays["success"][-1] else "failure"
        metadata = dict(format=FORMAT, episode_id=f"{self.manifest['session_id']}:{lane}:{ordinal}",
                        lane=lane, ordinal=ordinal, steps=len(rows), complete=complete, reason=reason,
                        outcome=outcome, reset_seed_input=buffer["seed"], seed_kind=buffer["seed_kind"],
                        task_id=self.manifest["task_id"], interface_id=self.manifest["interface_id"],
                        partition=self.manifest["partition"], behavior_segments=buffer["segments"],
                        index_sha256=self.index_sha, data_file=stem+".npz", sha256=hashlib.sha256(raw).hexdigest())
        index = _json_bytes(metadata)
        if self.saved_bytes + len(raw) + len(index) > MAX_TOTAL_BYTES:
            raise ValueError("Episode store total byte bound reached")
        _write(self.directory / (stem+".npz"), raw)
        _write(self.directory / (stem+".json"), index)
        self.saved_count += 1
        self.saved_bytes += len(raw) + len(index)
        self.ordinals[lane] += 1
        self.buffers[lane] = None
        return metadata["episode_id"]

    def finish(self, reason):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 256:
            raise ValueError("A nonempty finish reason <=256 characters is required")
        if self.closed:
            raise ValueError("Recorder is already finished")
        result = [self._persist(lane, reason, False) for lane, b in enumerate(self.buffers) if b is not None and b["rows"]]
        self.closed = True
        return result


def _read_file(path, limit):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Missing, symlinked, or oversized episode file")
    with path.open("rb") as handle:
        raw = handle.read(limit+1)
    if len(raw) > limit:
        raise ValueError("Oversized episode file")
    return raw


def _preflight_npz(raw, schema, expanded_limit):
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if len(entries) != len(schema) or {e.filename for e in entries} != {n+".npy" for n in schema} or sum(e.file_size for e in entries) > expanded_limit:
                raise ValueError("Unexpected or oversized NPZ members")
            for member in entries:
                if member.flag_bits & 1 or member.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                    raise ValueError("Unsafe NPZ member")
                name = member.filename[:-4]
                with archive.open(member) as handle:
                    version = np.lib.format.read_magic(handle)
                    reader = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}.get(version)
                    if reader is None:
                        raise ValueError("Unsupported NPY version")
                    position = handle.tell()
                    size = 2 if version == (1, 0) else 4
                    header_size = handle.read(size)
                    if len(header_size) != size or struct.unpack("<H" if size == 2 else "<I", header_size)[0] > 4096:
                        raise ValueError("Oversized NPY header")
                    handle.seek(position)
                    shape, fortran, dtype = reader(handle, max_header_size=4096)
                    expected_shape, expected_dtype = schema[name]
                    if shape != expected_shape or dtype != np.dtype(expected_dtype) or fortran or dtype.hasobject:
                        raise ValueError("Unsafe NPZ shape or dtype")
                    if member.file_size != handle.tell() + int(np.prod(shape))*dtype.itemsize:
                        raise ValueError("NPY declared size differs from numeric payload")
    except (OSError, EOFError, zipfile.BadZipFile, KeyError) as error:
        raise ValueError("Invalid episode NPZ") from error


def _safe_arrays(raw, length):
    schema = {name: ((length, *shape), bool if name in FLAGS else np.float32) for name, shape in ARRAYS.items()}
    _preflight_npz(raw, schema, MAX_FILE_BYTES)
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        return {name: loaded[name].copy() for name in ARRAYS}


def load_episodes(directory, expected_task_id, expected_interface_id, for_training=False):
    """Read bounded numeric episode dictionaries with a separate metadata entry.

    Returned keys are ``metadata`` and the eight arrays in ARRAYS. Loading an
    episode does not authorize relabeling its task or reusing another reward.
    """
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir() or type(for_training) is not bool:
        raise ValueError("Invalid episode directory or training flag")
    raw_index = _read_file(directory/"index.json", MAX_JSON_BYTES)
    manifest = _json_read(raw_index)
    if not isinstance(manifest, dict):
        raise ValueError("Invalid episode manifest")
    if manifest.get("format") != FORMAT or manifest.get("task_id") != _id(expected_task_id) or manifest.get("interface_id") != _id(expected_interface_id):
        raise ValueError("Episode task/interface mismatch")
    if manifest.get("partition") not in PARTITIONS or (for_training and manifest["partition"] != "training"):
        raise ValueError("Development/final episodes cannot become training data")
    count = manifest.get("num_envs")
    if type(count) is not int or not 1 <= count <= 4096 or type(manifest.get("auto_reset")) is not bool or not isinstance(manifest.get("session_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", manifest["session_id"]):
        raise ValueError("Invalid episode manifest")
    _behavior(manifest.get("behavior"))
    index_sha = hashlib.sha256(raw_index).hexdigest()
    files = list(directory.iterdir())
    if len(files) > 2*MAX_EPISODES+1:
        raise ValueError("Too many episode files")
    if any(p.is_symlink() or not p.is_file() for p in files) or sum(p.stat().st_size for p in files) > MAX_TOTAL_BYTES:
        raise ValueError("Invalid files or oversized episode store")
    indices = sorted(p for p in files if re.fullmatch(r"episode-\d{6}-\d{8}\.json", p.name))
    expected_files = {"index.json"} | {p.name for p in indices} | {p.with_suffix(".npz").name for p in indices}
    if {p.name for p in files} != expected_files:
        raise ValueError("Unexpected or orphaned episode files")
    result, total, ordinals = [], 0, [0]*count
    for path in indices:
        metadata = _json_read(_read_file(path, MAX_JSON_BYTES))
        if not isinstance(metadata, dict):
            raise ValueError("Invalid episode index entry")
        lane, ordinal, length = (metadata.get(k) for k in ("lane", "ordinal", "steps"))
        if type(lane) is not int or not 0 <= lane < count or type(ordinal) is not int or ordinal != ordinals[lane] or type(length) is not int or not 1 <= length <= MAX_STEPS:
            raise ValueError("Invalid lane, episode order or length")
        if path.stem != f"episode-{lane:06d}-{ordinal:08d}" or metadata.get("episode_id") != f"{manifest['session_id']}:{lane}:{ordinal}":
            raise ValueError("Episode identity/boundary mismatch")
        for key in ("task_id", "interface_id", "partition", "format"):
            if metadata.get(key) != manifest[key]:
                raise ValueError("Episode provenance mismatch")
        if metadata.get("index_sha256") != index_sha or metadata.get("data_file") != path.with_suffix(".npz").name:
            raise ValueError("Episode index binding mismatch")
        raw = _read_file(path.with_suffix(".npz"), MAX_FILE_BYTES)
        total += len(raw)
        if total > MAX_TOTAL_BYTES or hashlib.sha256(raw).hexdigest() != metadata.get("sha256"):
            raise ValueError("Episode digest or total-size mismatch")
        arrays = _safe_arrays(raw, length)
        if any(not np.isfinite(value).all() for value in arrays.values()) or (np.abs(arrays["actions"]) > 1).any():
            raise ValueError("Nonfinite episode or unbounded executed action")
        if not np.array_equal(arrays["observations"][1:], arrays["next_observations"][:-1]):
            raise ValueError("Episode continuity mismatch")
        boundary = arrays["terminated"] | arrays["truncated"]
        complete = metadata.get("complete")
        if type(complete) is not bool or boundary[:-1].any() or bool(boundary[-1]) != complete:
            raise ValueError("Episode boundary mismatch")
        expected_outcome = "incomplete" if not complete else "unknown" if not arrays["success_known"][-1] else "success" if arrays["success"][-1] else "failure"
        if metadata.get("outcome") != expected_outcome or not isinstance(metadata.get("reason"), str) or not metadata["reason"]:
            raise ValueError("Episode outcome/reason mismatch")
        if len(metadata["reason"]) > 256:
            raise ValueError("Episode reason exceeds bound")
        if complete:
            reason = "terminated_and_truncated" if arrays["terminated"][-1] and arrays["truncated"][-1] else "terminated" if arrays["terminated"][-1] else "truncated"
            if metadata["reason"] != reason:
                raise ValueError("Completed episode reason differs from boundary flags")
        seed, kind = metadata.get("reset_seed_input"), metadata.get("seed_kind")
        if kind not in {"explicit_lane_seed", "batched_reset_seed_argument", "autoreset_unknown"} or (seed is not None and (type(seed) is not int or not 0 <= seed < 2**63)) or (kind == "autoreset_unknown" and seed is not None):
            raise ValueError("Invalid reset-seed provenance")
        if (arrays["success"] & ~arrays["success_known"]).any():
            raise ValueError("Unknown success flags must remain unset")
        segments = metadata.get("behavior_segments")
        if not isinstance(segments, list) or not segments or not all(isinstance(s, dict) for s in segments) or segments[0].get("start") != 0:
            raise ValueError("Missing behavior provenance")
        previous = -1
        for segment in segments:
            start = segment.get("start")
            if type(start) is not int or not previous < start < length:
                raise ValueError("Behavior segment order differs")
            _behavior(segment.get("behavior"))
            previous = start
        ordinals[lane] += 1
        result.append(dict(metadata=metadata, **arrays))
    return result


def import_legacy_demos(source_directory, destination, task_id, interface_id, *,
                        expected_collector_sha256=LEGACY_COLLECTOR_SHA256):
    """Import every COMPLETE episode from the one pinned legacy collector.

    Reuses amp_v1.load_demos with its strict production reader, including its
    existing >=4-success requirement. This is not an all-failed legacy importer.
    Boundary flags are reconstructed from that collector's enforced full50,
    auto_reset=False contract; legacy files did not store these flags themselves.
    The immutable receipt is a sibling ``<destination>-import-receipt.json``.
    """
    started = time.monotonic()
    source, destination = Path(source_directory), Path(destination)
    if source.is_symlink() or not source.is_dir() or destination.resolve().is_relative_to(source.resolve()):
        raise ValueError("Legacy source must exist and destination must be outside it")
    if expected_collector_sha256 != LEGACY_COLLECTOR_SHA256:
        raise ValueError("Unsupported collector: cannot infer another collector's boundaries")
    receipt_path = destination.with_name(destination.name+"-import-receipt.json")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FileExistsError("Legacy import receipt already exists")
    metadata_raw = _read_file(source/"dataset.json", MAX_JSON_BYTES)
    metadata = _json_read(metadata_raw)
    if not isinstance(metadata, dict) or metadata.get("collector_source_sha256") != expected_collector_sha256:
        raise ValueError("Legacy collector digest differs from the reviewed boundary contract")
    episodes, count = metadata.get("completed_episodes"), metadata.get("num_envs")
    if type(episodes) is not int or not 1 <= episodes <= MAX_LEGACY_EPISODES or type(count) is not int or not 1 <= count <= 4096 or episodes % count:
        raise ValueError("Invalid legacy complete-batch count")
    raw = _read_file(source/"trajectories.npz", MAX_LEGACY_COMPRESSED_BYTES)
    schema = {"observations": ((episodes, 51, 48), np.float32),
              "actions": ((episodes, 50, 8), np.float32), "task_rewards": ((episodes, 50), np.float32),
              "final_success": ((episodes,), bool), "public_training_seed": ((episodes,), np.int64)}
    _preflight_npz(raw, schema, MAX_LEGACY_EXPANDED_BYTES)
    from research.discovery.amp_v1 import load_demos
    data = load_demos(source)  # Strict defaults; never enable local smoke data.
    if data.dataset_sha256 != hashlib.sha256(raw).hexdigest():
        raise ValueError("Legacy source changed during validation")
    with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
        seeds = arrays["public_training_seed"].copy()
    for offset in range(0, episodes, count):
        batch = seeds[offset:offset+count]
        if not (batch == batch[0]).all() or not 0 <= int(batch[0]) < 2**63:
            raise ValueError("Legacy batch must have one explicit scalar reset-seed argument")
    observations, actions, rewards, successes = (getattr(data, key).numpy() for key in
                                                ("observations", "actions", "task_rewards", "final_success"))
    # The existing BC reader returns all rows AND selection indices. Deliberately
    # ignore both train_episode_indices and validation_episode_indices here.
    provenance = dict(kind="pinned-legacy-complete-episode-import", legacy_format=metadata["format"],
                      legacy_dataset_sha256=data.dataset_sha256,
                      legacy_metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
                      collector_source_sha256=expected_collector_sha256,
                      teacher_actor_weights_sha256=metadata.get("teacher_actor_weights_sha256"),
                      original_episode_index="ordinal*num_envs+lane; no success selection")
    record = EpisodeRecorder(destination, task_id, interface_id, "training", provenance, count, auto_reset=False)
    for offset in range(0, episodes, count):
        chosen = slice(offset, offset+count)
        record.start(observations[chosen, 0], int(seeds[offset]))
        for step in range(50):
            record.step(observations[chosen, step], actions[chosen, step], observations[chosen, step+1],
                        rewards[chosen, step], np.zeros(count, dtype=bool),
                        np.full(count, step == 49, dtype=bool),
                        {"success": successes[chosen]} if step == 49 else {})
    record.finish("legacy_complete_import")
    # Bind EVERY immutable file, not only index.json or the old trajectories NPZ.
    file_digests = {path.name: hashlib.sha256(_read_file(path, MAX_FILE_BYTES)).hexdigest()
                    for path in sorted(destination.iterdir())}
    data_digest = hashlib.sha256(_json_bytes(file_digests)).hexdigest()
    if (_read_file(source/"dataset.json", MAX_JSON_BYTES) != metadata_raw or
            hashlib.sha256(_read_file(source/"trajectories.npz", MAX_LEGACY_COMPRESSED_BYTES)).hexdigest() != data.dataset_sha256):
        raise ValueError("Legacy source changed during import")
    receipt = dict(format="automanip-legacy-episode-import-v1", task_id=task_id, interface_id=interface_id,
                   source_directory=str(source.resolve()), destination=str(destination.resolve()),
                   receipt_path=str(receipt_path.resolve()), old_dataset_sha256=data.dataset_sha256,
                   source_metadata_sha256=hashlib.sha256(metadata_raw).hexdigest(),
                   collector_source_sha256=expected_collector_sha256,
                   total_imported_transitions=episodes*50, total_imported_episodes=episodes,
                   successes=int(successes.sum()), failures=int(episodes-successes.sum()),
                   new_environment_transitions=0, new_teacher_data=0,
                   legacy_reader_requires_at_least_four_successes=True,
                   source_incomplete_transition_rows_not_imported=metadata.get("incomplete_transition_rows"),
                   reconstruction_basis="Pinned collect_batch enforces auto_reset=False, all50 steps, no terminations, truncation only/all lanes at step49. Final-success labels attached only at step49. Original seed entries are scalar batch reset inputs, not expanded lane seeds.",
                   selection="All legacy completed episodes retained, including BC validation episodes and failures",
                   source_episode_mapping="original_index=ordinal*num_envs+lane",
                   data_digest=data_digest, immutable_file_sha256=file_digests,
                   prior_costs={key: metadata.get(key) for key in
                                ("teacher_prefix_transitions", "teacher_prefix_trainer_seconds", "teacher_prefix_dollars", "collection_wall_seconds", "collection_dollars")},
                   import_wall_seconds=time.monotonic()-started,
                   note="Source files unchanged. Digests provide data binding, not cryptographic identity/authentication. No simulator or collector executed.")
    _write(receipt_path, _json_bytes(receipt))
    return receipt
