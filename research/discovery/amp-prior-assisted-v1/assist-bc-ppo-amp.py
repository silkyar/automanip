from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

# ManiSkill specific imports
import mani_skill.envs
from mani_skill.utils import gym_utils
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

@dataclass
class Args:
    exp_name: Optional[str] = None
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=True`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "ManiSkill"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    capture_video: bool = True
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""
    evaluate: bool = False
    """if toggled, only runs evaluation with the given model checkpoint and saves the evaluation trajectories"""
    checkpoint: Optional[str] = None
    """path to a pretrained checkpoint file to start evaluation/training from"""

    # Algorithm specific arguments
    env_id: str = "PickCube-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 512
    """the number of parallel environments"""
    num_eval_envs: int = 8
    """the number of parallel evaluation environments"""
    partial_reset: bool = True
    """whether to let parallel environments reset upon termination instead of truncation"""
    eval_partial_reset: bool = False
    """whether to let parallel evaluation environments reset upon termination instead of truncation"""
    num_steps: int = 50
    """the number of steps to run in each environment per policy rollout"""
    num_eval_steps: int = 50
    """the number of steps to run in each evaluation environment during evaluation"""
    reconfiguration_freq: Optional[int] = None
    """how often to reconfigure the environment during training"""
    eval_reconfiguration_freq: Optional[int] = 1
    """for benchmarking purposes we want to reconfigure the eval environment each reset to ensure objects are randomized in some tasks"""
    control_mode: Optional[str] = "pd_joint_delta_pos"
    """the control mode to use for the environment"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.8
    """the discount factor gamma"""
    gae_lambda: float = 0.9
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = False
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = 0.1
    """the target KL divergence threshold"""
    reward_scale: float = 1.0
    """Scale the reward by this factor"""
    eval_freq: int = 25
    """evaluation frequency in terms of iterations"""
    save_train_video_freq: Optional[int] = None
    """frequency to save training videos in terms of iterations"""
    finite_horizon_gae: bool = False
    # BEGIN DISCOVERY_ASSIST arguments
    assist_dataset: str = ""
    assist_bc_actor: str = ""
    assist_bc_report: str = ""
    # END DISCOVERY_ASSIST arguments


    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1)),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01*np.sqrt(2)),
        )
        self.actor_logstd = nn.Parameter(torch.ones(1, np.prod(envs.single_action_space.shape)) * -0.5)

    def get_value(self, x):
        return self.critic(x)
    def get_action(self, x, deterministic=False):
        action_mean = self.actor_mean(x)
        if deterministic:
            return action_mean
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        return probs.sample()
    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

class Logger:
    def __init__(self, log_wandb=False, tensorboard: SummaryWriter = None) -> None:
        self.writer = tensorboard
        self.log_wandb = log_wandb
    def add_scalar(self, tag, scalar_value, step):
        if self.log_wandb:
            wandb.log({tag: scalar_value}, step=step)
        self.writer.add_scalar(tag, scalar_value, step)
    def close(self):
        self.writer.close()

# BEGIN AUTOMANIP schedule-function
def entropy_coefficient(step, total):
    """Stay at 0.003 through 15M transitions, then linearly reach zero."""
    if total<=15_000_000:return .003
    return .003*max(0.,min(1.,(total-step)/(total-15_000_000)))

# END AUTOMANIP schedule-function
# BEGIN PRECISION schedule
def refinement_schedule(step, method):
    fraction = max(0., min(1., (step - 25_600_000) / (49_971_200 - 25_600_000)))
    if method == 'precision-constant':
        return .003, .0003
    if method == 'precision-entropy-anneal':
        return .003 * (1 - fraction), .0003
    if method == 'precision-lr-anneal':
        return .003, .0003 * (1 - .9 * fraction)
    raise ValueError('unregistered precision recipe')

# END PRECISION schedule
# BEGIN DISCOVERY_ASSIST helpers
import hashlib, json
from pathlib import Path
TASK_FINGERPRINT = 'b524435b1d25e915ff9cbdc1cc18f129ded8257d12129f5a426bad9fa6e22d4f'
DATA_FORMAT = 'automanip-public-training-demos-v1'
PRODUCTION_PINS = {'mani_skill': '3.0.0b21', 'sapien': '3.0.0b1', 'torch': '2.6.0+cu124', 'numpy': '1.26.4', 'gymnasium': '0.29.1'}
AUTOMANIP_USE_PRIOR = True
class StudentActor(nn.Module):
    def __init__(self, hidden_sizes=(256, 256, 256)):
        super().__init__()
        if not 1 <= len(hidden_sizes) <= 3 or any(type(v) is not int or not 1 <= v <= 256 for v in hidden_sizes):
            raise ValueError("Unsupported public safe actor architecture")
        self.hidden_sizes = list(hidden_sizes)
        layers = []
        widths = [48, *hidden_sizes, 8]
        for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(hidden_sizes):
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, obs):
        return self.net(obs)

    def deterministic(self, obs):
        return self(obs).clamp(-1, 1)


def read_actor(directory, device="cpu"):
    """Candidate-side numeric reader; never substitutes for trusted validation."""
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    if (manifest["format"] != "automanip-json-f32-v1" or manifest["task_fingerprint"] != TASK_FINGERPRINT
            or manifest["activation"] != "tanh" or manifest["observation_dim"] != 48 or manifest["action_dim"] != 8):
        raise ValueError("Public actor contract mismatch")
    raw = (directory/"weights.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Actor digest mismatch")
    actor = StudentActor(tuple(manifest["hidden_sizes"]))
    data = json.loads(raw)
    expected = set()
    with torch.no_grad():
        for i, layer in enumerate(layer for layer in actor.net if isinstance(layer, nn.Linear)):
            for key in ("weight", "bias"):
                name = f"layers.{i}.{key}"
                expected.add(name)
                target = getattr(layer, key)
                entry = data[name]
                if entry["shape"] != list(target.shape):
                    raise ValueError("Actor tensor shape mismatch")
                value = torch.tensor(entry["values"], dtype=torch.float32).reshape(target.shape)
                if not torch.isfinite(value).all():
                    raise ValueError("Nonfinite actor tensor")
                target.copy_(value)
    if set(data) != expected:
        raise ValueError("Unexpected actor tensors")
    return actor.to(device), manifest


@dataclass
class DemoData:
    observations: torch.Tensor
    actions: torch.Tensor
    task_rewards: torch.Tensor
    final_success: torch.Tensor
    train_episode_indices: list[int]
    validation_episode_indices: list[int]
    metadata: dict
    dataset_sha256: str

    def training_pairs(self, device="cpu"):
        episodes = self.observations[self.train_episode_indices]
        return episodes[:, :-1].reshape(-1, 48).to(device), episodes[:, 1:].reshape(-1, 48).to(device)

    def behavior_rows(self, validation=False, device="cpu"):
        index = self.validation_episode_indices if validation else self.train_episode_indices
        return (self.observations[index, :-1].reshape(-1, 48).to(device),
                self.actions[index].reshape(-1, 8).to(device))


def load_demos(directory, split_seed=0, allow_local_cpu_smoke=False):
    directory = Path(directory)
    metadata = json.loads((directory/"dataset.json").read_text())
    raw_path = directory/"trajectories.npz"
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if metadata["format"] != DATA_FORMAT or metadata["sha256"] != digest:
        raise ValueError("Demo format or digest mismatch")
    expected = dict(env_id="StackCube-v1", observation_dim=48, action_dim=8, horizon=50,
                    control_mode="pd_joint_delta_pos", obs_mode="state", reward_mode="normalized_dense",
                    simulation_backend="physx_cuda", source_partition="public-training")
    if allow_local_cpu_smoke:
        expected["simulation_backend"] = "physx_cpu"
        if (metadata.get("purpose") != "local-cpu-dataflow-smoke" or metadata.get("comparison_eligible") is not False
                or metadata.get("versions", {}).get("sapien") != "3.0.4.dev20261009+7dba23f1"):
            raise ValueError("Explicit local CPU smoke provenance missing")
    elif any(metadata.get("versions", {}).get(k) != v for k, v in PRODUCTION_PINS.items()):
        raise ValueError("Production demonstration software pins mismatch")
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError("Demo public task or training-partition contract mismatch")
    with np.load(raw_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"observations", "actions", "task_rewards", "final_success", "public_training_seed"}:
            raise ValueError("Unexpected demo arrays")
        obs, actions, rewards = (arrays[k] for k in ("observations", "actions", "task_rewards"))
        success, seeds = arrays["final_success"], arrays["public_training_seed"]
        episodes = len(obs)
        if obs.shape != (episodes, 51, 48) or actions.shape != (episodes, 50, 8) or rewards.shape != (episodes, 50):
            raise ValueError("Demo episode shape mismatch")
        if success.shape != (episodes,) or success.dtype != np.bool_ or seeds.shape != (episodes,) or seeds.dtype != np.int64:
            raise ValueError("Demo outcome/seed dtype mismatch")
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in (obs, actions, rewards)):
            raise ValueError("Demo tensors must be finite float32")
        if np.max(np.abs(actions)) > 1.000001:
            raise ValueError("BC labels must be executed bounded teacher actions")
        if metadata["completed_episodes"] != episodes or metadata["successful_episodes"] != int(success.sum()):
            raise ValueError("Demo accounting mismatch")
        selected = np.flatnonzero(success).tolist()
        if len(selected) < 4:
            raise ValueError("At least4 successful training episodes required for BC split")
        ordered = sorted(selected, key=lambda i: hashlib.sha256(f"{digest}:{split_seed}:{i}".encode()).digest())
        validation_count = max(1, len(ordered)//10)
        validation, train = sorted(ordered[:validation_count]), sorted(ordered[validation_count:])
        return DemoData(torch.from_numpy(obs.copy()), torch.from_numpy(actions.copy()), torch.from_numpy(rewards.copy()),
                        torch.from_numpy(success.copy()), train, validation, metadata, digest)


def actual_successor(next_obs, done, infos):
    actual = next_obs.detach().clone()
    if done.bool().any():
        final = infos.get("final_observation")
        mask = infos.get("_final_observation")
        if final is None or mask is None or not torch.equal(mask.bool(), done.bool()) or final.shape != next_obs.shape:
            raise ValueError("AMP transition needs exact pre-reset final observations")
        actual[done.bool()] = final[done.bool()]
    if actual.shape[-1] != 48 or not torch.isfinite(actual).all():
        raise ValueError("Invalid AMP successor")
    return actual


def amp_style_reward(score):
    return (1-.25*(score.detach()-1).square()).clamp(0., 1.)


def mix_training_reward(task_reward, style_reward, weight=.25):
    if task_reward.shape != style_reward.shape or not 0 <= weight <= 1:
        raise ValueError("Training reward shape or declared weight invalid")
    if not torch.isfinite(task_reward).all() or not torch.isfinite(style_reward).all():
        raise ValueError("Nonfinite training reward")
    return task_reward.detach()+weight*style_reward.detach()


class TransitionDiscriminator(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("std", std.detach().clamp_min(.1).clone())
        with torch.random.fork_rng(devices=[]):
            self.net = nn.Sequential(nn.Linear(96, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def features(self, obs, next_obs):
        return torch.cat((((obs-self.mean)/self.std).clamp(-10, 10),
                          ((next_obs-self.mean)/self.std).clamp(-10, 10)), -1)

    def forward(self, features):
        return self.net(features).squeeze(-1)


def discriminator_loss(discriminator, expert_features, policy_features, gradient_penalty=10.):
    expert = expert_features.detach().requires_grad_(True)
    policy = policy_features.detach()
    expert_score, policy_score = discriminator(expert), discriminator(policy)
    gradient = torch.autograd.grad(expert_score.sum(), expert, create_graph=True)[0]
    penalty = gradient.square().sum(-1).mean()
    least_squares = (expert_score-1).square().mean()+(policy_score+1).square().mean()
    return least_squares+.5*gradient_penalty*penalty, dict(
        least_squares=float(least_squares.detach()), gradient_penalty=float(penalty.detach()),
        expert_score=float(expert_score.detach().mean()), policy_score=float(policy_score.detach().mean()))


class MotionPrior:
    def __init__(self, expert_obs, expert_next_obs, seed, device="cpu", capacity=100000):
        obs = expert_obs.detach().to(device)
        next_obs = expert_next_obs.detach().to(device)
        stats = torch.cat((obs, next_obs))
        self.discriminator = TransitionDiscriminator(stats.mean(0), stats.std(0, unbiased=False)).to(device)
        with torch.no_grad():
            self.expert = self.discriminator.features(obs, next_obs)
        self.optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.buffer = torch.empty(capacity, 96, device=device)
        self.capacity, self.position, self.size = capacity, 0, 0
        self.generator = torch.Generator(device=device).manual_seed(seed+27001)
        self.updates = self.sample_rows = self.transitions_added = 0

    @torch.no_grad()
    def reward_and_remember(self, obs, next_obs):
        features = self.discriminator.features(obs.detach(), next_obs.detach())
        reward = amp_style_reward(self.discriminator(features))
        count = len(features)
        if count > self.capacity:
            raise ValueError("Transition batch exceeds prior replay")
        index = (torch.arange(count, device=self.buffer.device)+self.position)%self.capacity
        self.buffer[index] = features
        self.position = (self.position+count)%self.capacity
        self.size = min(self.capacity, self.size+count)
        self.transitions_added += count
        return reward

    def update(self, rounds=32, batch_size=256):
        if self.size < 1:
            raise ValueError("Empty learner transition buffer")
        totals = {}
        for _ in range(rounds):
            expert_index = torch.randint(len(self.expert), (batch_size,), generator=self.generator, device=self.buffer.device)
            policy_index = torch.randint(self.size, (batch_size,), generator=self.generator, device=self.buffer.device)
            loss, stats = discriminator_loss(self.discriminator, self.expert[expert_index], self.buffer[policy_index])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite discriminator update")
            self.updates += 1
            self.sample_rows += 2*batch_size
            for key, value in dict(loss=float(loss.detach()), **stats).items():
                totals[key] = totals.get(key, 0.)+value/rounds
        return dict(totals, discriminator_updates=self.updates, discriminator_sample_rows=self.sample_rows,
                    learner_transition_rows=self.transitions_added, replay_rows=self.size)


def assist_initialize(agent, args, use_prior, device):
    if args.assist_bc_actor == "" or args.assist_dataset == "" or args.assist_bc_report == "":
        raise ValueError("Explicit public demo dataset and random-student BC artifact required")
    data = load_demos(args.assist_dataset)
    with torch.random.fork_rng(devices=[]):
        actor, manifest = read_actor(args.assist_bc_actor, device)
    if actor.hidden_sizes != [256, 256, 256]:
        raise ValueError("BC artifact must match pinned PPO actor architecture")
    bc_report_raw = Path(args.assist_bc_report).read_bytes()
    bc_report = json.loads(bc_report_raw)
    if (bc_report.get("teacher_weights_loaded") is not False or bc_report.get("student_initialization") != "random"
            or bc_report.get("comparison_eligible") is not True or bc_report.get("dataset_sha256") != data.dataset_sha256
            or bc_report.get("final", {}).get("weights_sha256") != manifest["weights_sha256"]):
        raise ValueError("BC provenance does not match a random-initialized, production-eligible student")
    agent.actor_mean.load_state_dict(actor.net.state_dict(), strict=True)
    prior = None
    if use_prior:
        obs, next_obs = data.training_pairs(device=device)
        prior = MotionPrior(obs, next_obs, args.seed, device=device)
    metadata = dict(
        track="prior-assisted random student: BC then PPO", teacher_weights_loaded=False,
        bc_report_sha256=hashlib.sha256(bc_report_raw).hexdigest(),
        bc_prefix_optimizer_updates=bc_report["optimizer_updates"],
        bc_prefix_demonstration_minibatch_rows=bc_report["demonstration_minibatch_rows"],
        bc_prefix_wall_seconds=bc_report["wall_seconds"],
        dataset_sha256=data.dataset_sha256, bc_student_weights_sha256=manifest["weights_sha256"],
        teacher_prefix_transitions=data.metadata["teacher_prefix_transitions"],
        teacher_generation_transitions=data.metadata["environment_transitions"],
        teacher_prefix_trainer_seconds=data.metadata["teacher_prefix_trainer_seconds"],
        teacher_prefix_dollars=data.metadata["teacher_prefix_dollars"],
        all_teacher_episodes=data.metadata["completed_episodes"],
        failed_teacher_episodes_preserved=data.metadata["failed_completed_episodes"],
        training_demo_episodes=len(data.train_episode_indices), validation_demo_episodes=len(data.validation_episode_indices),
        amp_enabled=use_prior, amp_style_weight=.25 if use_prior else 0.,
        task_reward_weight=1., discriminator_input="existing state48 followed by actual successor state48",
        discriminator_models_in_agent_serialization=False)
    print("AUTOMANIP_ASSISTANCE "+json.dumps(metadata, sort_keys=True), flush=True)
    return prior

# END DISCOVERY_ASSIST helpers
if __name__ == "__main__":
    args = tyro.cli(Args)
    # BEGIN DISCOVERY_ASSIST configuration
    if args.checkpoint is not None:
        raise ValueError("Assisted student cannot load teacher/native policy weights; use its random-initialized BC actor artifact")
    if args.env_id != "StackCube-v1" or args.control_mode != "pd_joint_delta_pos" or args.num_steps != 50 or args.partial_reset:
        raise ValueError("Assisted track requires the unchanged full-horizon public task/interface")
    # END DISCOVERY_ASSIST configuration
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.exp_name is None:
        args.exp_name = os.path.basename(__file__)[: -len(".py")]
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    else:
        run_name = args.exp_name


    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="physx_cuda")
    if args.control_mode is not None:
        env_kwargs["control_mode"] = args.control_mode
    envs = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, reconfiguration_freq=args.reconfiguration_freq, **env_kwargs)
    eval_envs = gym.make(args.env_id, num_envs=args.num_eval_envs, reconfiguration_freq=args.eval_reconfiguration_freq, **env_kwargs)
    if isinstance(envs.action_space, gym.spaces.Dict):
        envs = FlattenActionSpaceWrapper(envs)
        eval_envs = FlattenActionSpaceWrapper(eval_envs)
    if args.capture_video:
        eval_output_dir = f"runs/{run_name}/videos"
        if args.evaluate:
            eval_output_dir = f"{os.path.dirname(args.checkpoint)}/test_videos"
        print(f"Saving eval videos to {eval_output_dir}")
        if args.save_train_video_freq is not None:
            save_video_trigger = lambda x : (x // args.num_steps) % args.save_train_video_freq == 0
            envs = RecordEpisode(envs, output_dir=f"runs/{run_name}/train_videos", save_trajectory=False, save_video_trigger=save_video_trigger, max_steps_per_video=args.num_steps, video_fps=30)
        eval_envs = RecordEpisode(eval_envs, output_dir=eval_output_dir, save_trajectory=args.evaluate, trajectory_name="trajectory", max_steps_per_video=args.num_eval_steps, video_fps=30)
    envs = ManiSkillVectorEnv(envs, args.num_envs, ignore_terminations=not args.partial_reset, record_metrics=True)
    eval_envs = ManiSkillVectorEnv(eval_envs, args.num_eval_envs, ignore_terminations=not args.eval_partial_reset, record_metrics=True)
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_episode_steps = gym_utils.find_max_episode_steps_value(envs._env)
    logger = None
    if not args.evaluate:
        print("Running training")
        if args.track:
            import wandb
            config = vars(args)
            config["env_cfg"] = dict(**env_kwargs, num_envs=args.num_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=args.partial_reset)
            config["eval_env_cfg"] = dict(**env_kwargs, num_envs=args.num_eval_envs, env_id=args.env_id, reward_mode="normalized_dense", env_horizon=max_episode_steps, partial_reset=False)
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=False,
                config=config,
                name=run_name,
                save_code=True,
                group="PPO",
                tags=["ppo", "walltime_efficient"]
            )
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        logger = Logger(log_wandb=args.track, tensorboard=writer)
    else:
        print("Running evaluation")

    agent = Agent(envs).to(device)
    # BEGIN DISCOVERY_ASSIST initialize
    automanip_prior = assist_initialize(agent, args, AUTOMANIP_USE_PRIOR, device)
    # END DISCOVERY_ASSIST initialize
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # BEGIN AUTOMANIP deadline
    automanip_deadline = float(os.environ.get("AUTOMANIP_TRAINING_DEADLINE", time.monotonic() + 1300))
    # END AUTOMANIP deadline
    next_obs, _ = envs.reset(seed=args.seed)
    eval_obs, _ = eval_envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs, device=device)
    print(f"####")
    print(f"args.num_iterations={args.num_iterations} args.num_envs={args.num_envs} args.num_eval_envs={args.num_eval_envs}")
    print(f"args.minibatch_size={args.minibatch_size} args.batch_size={args.batch_size} args.update_epochs={args.update_epochs}")
    print(f"####")
    action_space_low, action_space_high = torch.from_numpy(envs.single_action_space.low).to(device), torch.from_numpy(envs.single_action_space.high).to(device)
    def clip_action(action: torch.Tensor):
        return torch.clamp(action.detach(), action_space_low, action_space_high)

    if args.checkpoint:
        agent.load_state_dict(torch.load(args.checkpoint))

    # BEGIN PRECISION counter-init
    automanip_optimizer_steps = 0
    # END PRECISION counter-init
    for iteration in range(1, args.num_iterations + 1):
        # BEGIN AUTOMANIP loop-budget-schedule
        if time.monotonic() >= automanip_deadline:
            print("TRAINING_TIME_BUDGET_REACHED", global_step, flush=True)
            break
        # END AUTOMANIP loop-budget-schedule
        # BEGIN PRECISION apply-schedule
        args.ent_coef, automanip_lr = refinement_schedule(global_step + args.batch_size, 'precision-constant')
        for automanip_group in optimizer.param_groups:
            automanip_group["lr"] = automanip_lr
        # END PRECISION apply-schedule
        print(f"Epoch: {iteration}, global_step={global_step}")
        final_values = torch.zeros((args.num_steps, args.num_envs), device=device)
        agent.eval()
        if iteration % args.eval_freq == 1:
            print("Evaluating")
            eval_obs, _ = eval_envs.reset()
            eval_metrics = defaultdict(list)
            num_episodes = 0
            for _ in range(args.num_eval_steps):
                with torch.no_grad():
                    eval_obs, eval_rew, eval_terminations, eval_truncations, eval_infos = eval_envs.step(agent.get_action(eval_obs, deterministic=True))
                    if "final_info" in eval_infos:
                        mask = eval_infos["_final_info"]
                        num_episodes += mask.sum()
                        for k, v in eval_infos["final_info"]["episode"].items():
                            eval_metrics[k].append(v)
            print(f"Evaluated {args.num_eval_steps * args.num_eval_envs} steps resulting in {num_episodes} episodes")
            for k, v in eval_metrics.items():
                mean = torch.stack(v).float().mean()
                if logger is not None:
                    logger.add_scalar(f"eval/{k}", mean, global_step)
                print(f"eval_{k}_mean={mean}")
            if args.evaluate:
                break
        if args.save_model and iteration % args.eval_freq == 1:
            model_path = f"runs/{run_name}/ckpt_{iteration}.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        rollout_time = time.time()
        # BEGIN DISCOVERY_ASSIST rollout-counter
        automanip_style_mean = 0.
        # END DISCOVERY_ASSIST rollout-counter
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(clip_action(action))
            next_done = torch.logical_or(terminations, truncations).to(torch.float32)
            rewards[step] = reward.view(-1) * args.reward_scale
            # BEGIN DISCOVERY_ASSIST train-only-reward
            if automanip_prior is not None:
                automanip_actual_next = actual_successor(next_obs, next_done, infos)
                automanip_style = automanip_prior.reward_and_remember(obs[step], automanip_actual_next)
                rewards[step] = mix_training_reward(rewards[step], automanip_style*args.reward_scale, .25)
                automanip_style_mean += float(automanip_style.mean())/args.num_steps
            # END DISCOVERY_ASSIST train-only-reward

            if "final_info" in infos:
                final_info = infos["final_info"]
                done_mask = infos["_final_info"]
                for k, v in final_info["episode"].items():
                    logger.add_scalar(f"train/{k}", v[done_mask].float().mean(), global_step)
                with torch.no_grad():
                    final_values[step, torch.arange(args.num_envs, device=device)[done_mask]] = agent.get_value(infos["final_observation"][done_mask]).view(-1)
        rollout_time = time.time() - rollout_time
        # BEGIN DISCOVERY_ASSIST discriminator-update
        if automanip_prior is not None:
            automanip_prior_started = time.time()
            automanip_prior_stats = automanip_prior.update(rounds=32, batch_size=256)
            automanip_prior_stats["discriminator_update_seconds"] = time.time()-automanip_prior_started
            automanip_prior_stats["mean_style_reward"] = automanip_style_mean
            for automanip_key, automanip_value in automanip_prior_stats.items():
                logger.add_scalar("audit/amp/"+automanip_key, automanip_value, global_step)
            print("AUTOMANIP_PRIOR_DIAGNOSTIC "+json.dumps(automanip_prior_stats, sort_keys=True), flush=True)
        # END DISCOVERY_ASSIST discriminator-update
        # BEGIN AUTOMANIP action-audit
        logger.add_scalar("audit/sampled_action_clip_fraction", (actions.abs() > 1).float().mean().item(), global_step)
        for automanip_dim in range(actions.shape[-1]):
            logger.add_scalar(f"audit/action_clip_dim_{automanip_dim}", (actions[..., automanip_dim].abs() > 1).float().mean().item(), global_step)
        # END AUTOMANIP action-audit
        # bootstrap value according to termination and truncation
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_not_done = 1.0 - next_done
                    nextvalues = next_value
                else:
                    next_not_done = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                real_next_values = next_not_done * nextvalues + final_values[t] # t instead of t+1
                # next_not_done means nextvalues is computed from the correct next_obs
                # if next_not_done is 1, final_values is always 0
                # if next_not_done is 0, then use final_values, which is computed according to bootstrap_at_done
                if args.finite_horizon_gae:
                    """
                    See GAE paper equation(16) line 1, we will compute the GAE based on this line only
                    1             *(  -V(s_t)  + r_t                                                               + gamma * V(s_{t+1})   )
                    lambda        *(  -V(s_t)  + r_t + gamma * r_{t+1}                                             + gamma^2 * V(s_{t+2}) )
                    lambda^2      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2}                         + ...                  )
                    lambda^3      *(  -V(s_t)  + r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + gamma^3 * r_{t+3}
                    We then normalize it by the sum of the lambda^i (instead of 1-lambda)
                    """
                    if t == args.num_steps - 1: # initialize
                        lam_coef_sum = 0.
                        reward_term_sum = 0. # the sum of the second term
                        value_term_sum = 0. # the sum of the third term
                    lam_coef_sum = lam_coef_sum * next_not_done
                    reward_term_sum = reward_term_sum * next_not_done
                    value_term_sum = value_term_sum * next_not_done

                    lam_coef_sum = 1 + args.gae_lambda * lam_coef_sum
                    reward_term_sum = args.gae_lambda * args.gamma * reward_term_sum + lam_coef_sum * rewards[t]
                    value_term_sum = args.gae_lambda * args.gamma * value_term_sum + args.gamma * real_next_values

                    advantages[t] = (reward_term_sum + value_term_sum) / lam_coef_sum - values[t]
                else:
                    delta = rewards[t] + args.gamma * real_next_values - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * next_not_done * lastgaelam # Here actually we should use next_not_terminated, but we don't have lastgamlam if terminated
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        update_time = time.time()
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                # BEGIN AUTOMANIP gradient-audit
                if epoch == 0 and start == 0:
                    automanip_actor = list(agent.actor_mean.parameters()) + [agent.actor_logstd]
                    automanip_ag = sum(p.grad.detach().square().sum() for p in automanip_actor if p.grad is not None).sqrt().item()
                    automanip_cg = sum(p.grad.detach().square().sum() for p in agent.critic.parameters() if p.grad is not None).sqrt().item()
                    logger.add_scalar("audit/actor_gradient_norm", automanip_ag, global_step)
                    logger.add_scalar("audit/critic_gradient_norm", automanip_cg, global_step)
                    logger.add_scalar("audit/global_clip_multiplier", min(1., args.max_grad_norm / (np.hypot(automanip_ag, automanip_cg) + 1e-6)), global_step)
                # END AUTOMANIP gradient-audit
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                # BEGIN PRECISION counter-step
                automanip_optimizer_steps += 1
                # END PRECISION counter-step

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        update_time = time.time() - update_time

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # BEGIN PRECISION accounting
        logger.add_scalar("audit/optimizer_steps", automanip_optimizer_steps, global_step)
        logger.add_scalar("audit/training_entropy_coef", args.ent_coef, global_step)
        for automanip_dim in range(agent.actor_logstd.shape[1]):
            logger.add_scalar(f"audit/logstd_dim_{automanip_dim}", agent.actor_logstd[0, automanip_dim].detach().item(), global_step)
        # END PRECISION accounting
        logger.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        logger.add_scalar("losses/value_loss", v_loss.item(), global_step)
        logger.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        logger.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        logger.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        logger.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        logger.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        logger.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        logger.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        logger.add_scalar("time/step", global_step, global_step)
        logger.add_scalar("time/update_time", update_time, global_step)
        logger.add_scalar("time/rollout_time", rollout_time, global_step)
        logger.add_scalar("time/rollout_fps", args.num_envs * args.num_steps / rollout_time, global_step)
    if not args.evaluate:
        if args.save_model:
            model_path = f"runs/{run_name}/final_ckpt.pt"
            torch.save(agent.state_dict(), model_path)
            print(f"model saved to {model_path}")
        logger.close()
    envs.close()
    eval_envs.close()
