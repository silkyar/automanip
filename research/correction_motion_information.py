"""Pre-decision features only; no labels, task flags or future frames accepted."""
import numpy as np


def quaternion_velocity(previous, current, frequency=20.):
    previous, current = [np.asarray(x, dtype=np.float64).copy() for x in (previous, current)]
    if previous.shape != current.shape or previous.ndim != 2 or previous.shape[1] != 4:
        raise ValueError('Expected batch of scalar-first pose quaternions')
    for x in (previous, current):
        norm = np.linalg.norm(x, axis=-1, keepdims=True)
        if not np.isfinite(x).all() or (norm < 1e-8).any():
            raise ValueError('Invalid quaternion')
        x /= norm
    # q_current * conjugate(q_previous), expressed in world coordinates.
    w = current[:, :1]*previous[:, :1]+(current[:, 1:]*previous[:, 1:]).sum(-1, keepdims=True)
    v = -current[:, :1]*previous[:, 1:]+previous[:, :1]*current[:, 1:]-np.cross(current[:, 1:], previous[:, 1:])
    sign = np.where(w < 0, -1, 1)
    w, v = w*sign, v*sign
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2*np.arctan2(norm, np.maximum(w, 0))
    factor = np.divide(angle, norm, out=np.full_like(norm, 2.), where=norm > 1e-10)
    return v*factor*frequency


def extract(observations, linear_velocity, angular_velocity, decision):
    observations = np.asarray(observations)
    linear_velocity, angular_velocity = np.asarray(linear_velocity), np.asarray(angular_velocity)
    decision = np.asarray(decision)
    if (observations.shape != (51, 64, 48) or linear_velocity.shape != (50, 64, 3)
            or angular_velocity.shape != (50, 64, 3) or decision.shape != (64,)
            or not np.issubdtype(decision.dtype, np.integer)
            or not np.isin(decision, np.arange(10,45)).all()):
        raise ValueError('Registered recorded-data layout differs')
    lanes = np.arange(64)
    current = observations[decision, lanes].astype(np.float64).copy()
    previous = observations[decision-1, lanes].astype(np.float64).copy()
    oracle = np.column_stack([linear_velocity[decision-1, lanes], angular_velocity[decision-1, lanes]])
    history = np.column_stack([(current[:,25:28]-previous[:,25:28])*20,
                               quaternion_velocity(previous[:,28:32], current[:,28:32])])
    state = np.column_stack([current, decision/50])
    if not all(np.isfinite(x).all() for x in (state, oracle, history)):
        raise ValueError('Nonfinite pre-intervention features')
    return dict(current=state, oracle=oracle, history=history)


def time_basis(decision):
    bins = np.minimum((np.asarray(decision)-10)//5, 6)
    if not np.isin(bins, np.arange(7)).all():
        raise ValueError('Decision outside registered time bins')
    return np.eye(7)[bins]


def shuffle_motion(motion, batch, decision, seed):
    result = np.asarray(motion).copy()
    rng = np.random.default_rng(seed)
    for group in np.unique(batch):
        for time_bin in range(7):
            rows = np.flatnonzero((batch==group) & (time_basis(decision)[:,time_bin] == 1))
            result[rows] = result[rng.permutation(rows)]
    return result


def fit_map(train, all_rows, seed, width):
    """Scale using fitting rows, then a single fixed random Fourier draw."""
    mean = train.mean(0)
    scale = np.maximum(train.std(0), .001)
    rng = np.random.default_rng(seed)
    weight = rng.standard_normal((width, train.shape[1])).T/np.sqrt(train.shape[1])
    bias = np.random.default_rng(seed+1).uniform(0, 2*np.pi, width)
    transformed = np.sqrt(2/width)*np.cos(((all_rows-mean)/scale)@weight+bias)
    return transformed, dict(mean=mean, scale=scale, weight=weight, bias=bias)


def ridge(design, target):
    gram = design.T@design/len(design)
    penalty = .001*np.trace(gram)/len(gram)
    target_product = design.T@target/len(design)
    coefficient = np.linalg.solve(gram+penalty*np.eye(len(gram)),target_product)
    if np.max(np.abs((gram+penalty*np.eye(len(gram)))@coefficient-target_product)) > 1e-9:
        raise ValueError('Probability normal equation residual too large')
    return coefficient


def contrast(control_errors, candidate_errors, batch):
    groups = np.unique(batch)
    if groups.shape != (8,):
        raise ValueError('Eight held vector batches required')
    differences = np.array([(control_errors[batch==g]-candidate_errors[batch==g]).mean() for g in groups])
    indices = np.random.default_rng(97109999).integers(0,8,(10000,8))
    return dict(mean_brier_gain=float(differences.mean()),
        interval_99_375=np.quantile(differences[indices].mean(1),[.003125,.996875]).tolist(),
        paired_block_gains=differences.tolist())
