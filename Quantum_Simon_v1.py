# Quantum-10-Simon



"""
Quantum_Simon.py

Python 3.11
Qiskit 1.4.4

IZVORNI SIMON
------------
Dva registra sa po sest kubita.
H -> U_f -> H.
f_b(x) = x XOR (bit_p(x) * b), gde je p izabrani bit perioda b.
Za b != 0:
    f_b(x) = f_b(y) ako i samo ako y pripada {x, x XOR b}.

Iz izlazne raspodele prvog registra dobijaju se jednacine:
    z . b = 0 mod 2.

Period se rekonstruise Gaussovom eliminacijom nad GF(2).
Koristi se tacna raspodela simulatora, bez slucajnih merenja.

DOPUNA ZA LOTO
--------------
XOR period je hipoteza prediktivnog modela.
Ne predstavlja potvrdjen vremenski ciklus izvucenih brojeva.

Za svaku sortiranu poziciju uci se raspodela prelaza:
    d = sledeci_broj_kod XOR prethodni_broj_kod
    broj_kod = broj - 1.

Amplitudni model je invarijantan na d -> d XOR b.
Svaki Simonov par ima naucenu ukupnu masu.
Pet RY uglova i CNOT lanac odredjuju amplitude predstavnika.
H i CNOT operacije zatim formiraju Simonove parove.

Simonov algoritam proverava period konstruisanog orakla.
Ne tvrdi se da sam Simonov algoritam predvidja sledece kolo.

KLASICNI DEO
------------
L-BFGS-B optimizuje uglove kvantnog kola.
Brier gubitak predstavlja regresiju raspodele.
Dinamicko programiranje daje tacan globalni MAP legalne kombinacije.

Podela je hronoloska.
Test ne ucestvuje u izboru modela.
Zavrsno ucenje koristi sve istorijske prelaze.
"""

import csv
import hashlib
import itertools
import random
from pathlib import Path

import numpy as np
import qiskit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from scipy.special import logsumexp


random.seed(39)
np.random.seed(39)


FILES = (
    Path("/data/loto7_4716_k89.csv"),
    Path("/data/loto7_4716_k89_loto_2980.csv"),
    Path("/data/loto7_4716_k89_loto_plus_1736.csv"),
)

N = 39
K = 7
QUBITS = 6
DIM = 1 << QUBITS
QUOTIENT_QUBITS = QUBITS - 1
QUOTIENT_DIM = 1 << QUOTIENT_QUBITS

REGULARIZATIONS = (0.0001, 0.001)
PRESELECT_PERIODS = 3

TRAIN_FRACTION = 0.70
VALIDATION_END_FRACTION = 0.85

MAX_ITERATIONS = 200
GRADIENT_TOLERANCE = 1e-6
RESTARTS = 2

INDICES = np.arange(DIM, dtype=np.int64)
PARITY = np.array(
    [value.bit_count() & 1 for value in range(DIM)],
    dtype=np.int64,
)
HADAMARD = (
    1.0
    - 2.0 * PARITY[
        np.bitwise_and(INDICES[:, None], INDICES[None, :])
    ]
) / np.sqrt(DIM)

QUOTIENT_BITS = (
    (
        np.arange(QUOTIENT_DIM, dtype=np.int64)[:, None]
        >> np.arange(QUOTIENT_QUBITS, dtype=np.int64)[None, :]
    )
    & 1
)


class DistributionError(ArithmeticError):
    pass


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load_csv(path):
    raw = path.read_bytes()
    reader = csv.reader(raw.decode("utf-8-sig").splitlines())
    rows = []

    for line_number, row in enumerate(reader, start=1):
        if not row or all(not cell.strip() for cell in row):
            continue

        require(
            len(row) == K,
            f"{path.name}, red {line_number}: potrebno je sedam kolona.",
        )

        try:
            values = [int(cell.strip()) for cell in row]
        except ValueError as exc:
            raise RuntimeError(
                f"{path.name}, red {line_number}: necelobrojna vrednost."
            ) from exc

        require(
            all(1 <= value <= N for value in values),
            f"{path.name}, red {line_number}: broj van opsega.",
        )
        require(
            len(set(values)) == K,
            f"{path.name}, red {line_number}: ponovljen broj.",
        )
        require(
            values == sorted(values),
            f"{path.name}, red {line_number}: brojevi nisu rastuci.",
        )
        rows.append(values)

    require(
        len(rows) >= 5,
        f"{path.name}: nedovoljno istorijskih kola.",
    )

    return {
        "path": path,
        "data": np.asarray(rows, dtype=np.int64) - 1,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def pivot_bit(period):
    require(0 < period < DIM, "Neispravan nenulti XOR period.")
    return (period & -period).bit_length() - 1


def oracle_values(period):
    require(0 <= period < DIM, "Period je van opsega.")

    if period == 0:
        return INDICES.copy()

    pivot = pivot_bit(period)
    return np.bitwise_xor(
        INDICES,
        ((INDICES >> pivot) & 1) * period,
    )


def simon_circuit(period):
    """Izvorno Simonovo kolo, bez merenja."""
    require(0 <= period < DIM, "Neispravan Simonov period.")

    circuit = QuantumCircuit(2 * QUBITS)
    circuit.h(range(QUBITS))

    for qubit in range(QUBITS):
        circuit.cx(qubit, QUBITS + qubit)

    if period != 0:
        pivot = pivot_bit(period)

        for qubit in range(QUBITS):
            if (period >> qubit) & 1:
                circuit.cx(pivot, QUBITS + qubit)

    circuit.h(range(QUBITS))
    return circuit


def gf2_nullspace(rows):
    """RREF i baza nul-prostora nad GF(2), sa celobrojnim redovima."""
    matrix = [int(row) for row in rows if int(row) != 0]
    pivot_columns = []
    rank = 0

    for column in range(QUBITS):
        selected = next(
            (
                row_index
                for row_index in range(rank, len(matrix))
                if (matrix[row_index] >> column) & 1
            ),
            None,
        )

        if selected is None:
            continue

        matrix[rank], matrix[selected] = (
            matrix[selected],
            matrix[rank],
        )

        for row_index in range(len(matrix)):
            if (
                row_index != rank
                and ((matrix[row_index] >> column) & 1)
            ):
                matrix[row_index] ^= matrix[rank]

        pivot_columns.append(column)
        rank += 1

    free_columns = [
        column
        for column in range(QUBITS)
        if column not in pivot_columns
    ]

    basis = []

    for free_column in free_columns:
        vector = 1 << free_column

        for row_index, column in enumerate(pivot_columns):
            if (matrix[row_index] >> free_column) & 1:
                vector |= 1 << column

        basis.append(vector)

    for vector in basis:
        require(
            all((row & vector).bit_count() % 2 == 0 for row in rows),
            "GF(2) resenje ne zadovoljava jednacine.",
        )

    return rank, basis


def recover_period(probabilities):
    """
    Koristi se tacna podrska izlazne raspodele.
    Prag odvaja numericku gresku od teorijskih masa 1/32 ili 1/64.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    require(
        probabilities.shape == (DIM,),
        "Neispravan oblik Simonove raspodele.",
    )
    require(
        np.isclose(probabilities.sum(), 1.0, atol=2e-10),
        "Simonova raspodela nije normalizovana.",
    )

    rows = np.flatnonzero(probabilities > 1e-10).tolist()
    rank, basis = gf2_nullspace(rows)

    if rank == QUBITS:
        require(len(basis) == 0, "Neispravan nul-prostor.")
        return 0

    require(
        rank == QUBITS - 1 and len(basis) == 1,
        "Simonova podrska ne odredjuje jedinstven nenulti period.",
    )
    return int(basis[0])


def verify_simon(period):
    values = oracle_values(period)

    for x in range(DIM):
        collisions = np.flatnonzero(values == values[x])

        expected = (
            [x]
            if period == 0
            else sorted([x, x ^ period])
        )

        require(
            collisions.tolist() == expected,
            "Orakl ne ispunjava Simonovo obecanje.",
        )

    circuit = simon_circuit(period)
    actual = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=np.complex128,
    ).reshape(DIM, DIM)

    # Red je drugi registar, kolona je prvi registar.
    reference = np.zeros((DIM, DIM), dtype=np.complex128)
    reference[values, INDICES] = 1.0 / np.sqrt(DIM)
    reference = reference @ HADAMARD.T

    require(
        np.allclose(actual, reference, atol=2e-10, rtol=2e-10),
        "Qiskit i referentno Simonovo kolo se razlikuju.",
    )

    probabilities = np.sum(np.abs(actual) ** 2, axis=0)
    recovered = recover_period(probabilities)

    require(
        recovered == period,
        "GF(2) rekonstrukcija nije vratila period orakla.",
    )

    if period != 0:
        support = probabilities > 1e-10
        require(
            np.all(PARITY[np.bitwise_and(INDICES[support], period)] == 0),
            "Simonova podrska nije ortogonalna periodu.",
        )

    return recovered


def make_structure(period):
    """
    Pet baznih bitova -> CNOT lanac -> predstavnik sa pivot bitom 0.
    Svaki predstavnik se zatim siri na par {r, r XOR period}.
    """
    pivot = pivot_bit(period)
    others = [bit for bit in range(QUBITS) if bit != pivot]

    assignment = np.full(DIM, -1, dtype=np.int64)

    for raw in range(QUOTIENT_DIM):
        transformed = raw

        for bit in range(QUOTIENT_QUBITS - 1):
            if (transformed >> bit) & 1:
                transformed ^= 1 << (bit + 1)

        representative = 0

        for bit, physical_bit in enumerate(others):
            representative |= (
                ((transformed >> bit) & 1) << physical_bit
            )

        assignment[representative] = raw
        assignment[representative ^ period] = raw

    require(
        np.all(assignment >= 0),
        "Simonovi parovi ne pokrivaju svih 64 stanja.",
    )
    require(
        np.all(np.bincount(assignment, minlength=QUOTIENT_DIM) == 2),
        "Predstavnik nema tacno dva clana.",
    )
    require(
        np.array_equal(
            assignment,
            assignment[np.bitwise_xor(INDICES, period)],
        ),
        "Kodiranje parova nije XOR invarijantno.",
    )

    return {
        "period": period,
        "pivot": pivot,
        "others": others,
        "assignment": assignment,
    }


def quotient_state_and_derivatives(theta):
    theta = np.asarray(theta, dtype=np.float64)
    require(
        theta.shape == (QUOTIENT_QUBITS,),
        "Neispravan broj RY uglova.",
    )
    require(np.isfinite(theta).all(), "Nekonacni RY uglovi.")

    half = theta / 2.0
    cosine = np.cos(half)
    sine = np.sin(half)

    factors = np.where(
        QUOTIENT_BITS == 0,
        cosine[None, :],
        sine[None, :],
    )
    amplitudes = np.prod(factors, axis=1)

    squared_factors = factors ** 2
    probabilities = np.prod(squared_factors, axis=1)

    local_derivatives = np.where(
        QUOTIENT_BITS == 0,
        -0.5 * np.sin(theta)[None, :],
        0.5 * np.sin(theta)[None, :],
    )

    derivatives = np.empty(
        (QUOTIENT_DIM, QUOTIENT_QUBITS),
        dtype=np.float64,
    )

    for bit in range(QUOTIENT_QUBITS):
        remaining = [
            other
            for other in range(QUOTIENT_QUBITS)
            if other != bit
        ]
        derivatives[:, bit] = (
            np.prod(squared_factors[:, remaining], axis=1)
            * local_derivatives[:, bit]
        )

    require(
        np.isclose(probabilities.sum(), 1.0, atol=2e-12),
        "Raspodela predstavnika nije normalizovana.",
    )

    return amplitudes, probabilities, derivatives


def model_state(theta, structure, previous=0):
    amplitudes, _, _ = quotient_state_and_derivatives(theta)
    base = amplitudes[structure["assignment"]] / np.sqrt(2.0)

    require(0 <= previous < DIM, "Neispravan prethodni ulaz.")
    return base[np.bitwise_xor(INDICES, previous)]


def amplitude_logs(amplitudes):
    absolute = np.abs(amplitudes)
    result = np.full(absolute.shape, -np.inf, dtype=np.float64)
    positive = absolute > 0.0
    result[positive] = 2.0 * np.log(absolute[positive])
    return result


def forecast_qiskit(theta, structure, previous):
    circuit = QuantumCircuit(QUBITS)
    others = structure["others"]
    pivot = structure["pivot"]
    period = structure["period"]

    for index, qubit in enumerate(others):
        circuit.ry(float(theta[index]), qubit)

    for index in range(len(others) - 1):
        circuit.cx(others[index], others[index + 1])

    circuit.h(pivot)

    for qubit in others:
        if (period >> qubit) & 1:
            circuit.cx(pivot, qubit)

    for qubit in range(QUBITS):
        if (previous >> qubit) & 1:
            circuit.x(qubit)

    actual = np.asarray(
        Statevector.from_instruction(circuit).data,
        dtype=np.complex128,
    )
    expected = model_state(theta, structure, previous)

    require(
        np.allclose(actual, expected, atol=2e-10, rtol=2e-10),
        "Prediktivno Qiskit kolo daje razlicite amplitude.",
    )
    require(
        np.isclose(np.vdot(actual, actual).real, 1.0, atol=2e-10),
        "Prediktivno stanje nije normalizovano.",
    )

    return actual


def transition_counts(data, end):
    require(1 <= end < len(data), "Neispravna granica treninga.")

    transitions = np.bitwise_xor(data[1:end + 1], data[:end])
    counts = np.zeros((K, DIM), dtype=np.float64)

    for slot in range(K):
        np.add.at(counts[slot], transitions[:, slot], 1.0)

    require(
        np.all(counts.sum(axis=1) == end),
        "Nisu obuhvaceni svi istorijski prelazi.",
    )

    return counts


def objective(theta, counts, structure, regularization):
    _, quotient_probabilities, quotient_derivatives = (
        quotient_state_and_derivatives(theta)
    )

    assignment = structure["assignment"]
    probabilities = quotient_probabilities[assignment] / 2.0
    derivatives = quotient_derivatives[assignment] / 2.0

    count_sum = float(np.sum(counts))
    require(count_sum > 0.0, "Prazan skup prelaza.")
    targets = counts / count_sum

    loss = (
        np.dot(probabilities, probabilities)
        - 2.0 * np.dot(targets, probabilities)
        + 1.0
    )
    gradient = 2.0 * derivatives.T @ (probabilities - targets)

    loss += regularization * np.dot(theta, theta)
    gradient += 2.0 * regularization * theta

    require(
        np.isfinite(loss) and np.isfinite(gradient).all(),
        "Nekonacan gubitak ili gradijent.",
    )

    return float(loss), gradient


def initial_parameters(restart):
    index = np.arange(1, QUOTIENT_QUBITS + 1, dtype=np.float64)

    if restart == 0:
        return 0.7 + 0.2 * np.sin(index)

    return 1.6 + 0.3 * np.cos(index * np.sqrt(2.0))


def fit_models(counts, structure, regularization):
    models = []
    diagnostics = []

    for slot in range(K):
        best_loss = np.inf
        best_theta = None
        best_status = ""

        for restart in range(RESTARTS):
            tracked = {"loss": np.inf, "theta": None}

            def tracked_objective(theta):
                loss, gradient = objective(
                    theta, counts[slot], structure, regularization
                )

                if loss < tracked["loss"]:
                    tracked["loss"] = loss
                    tracked["theta"] = theta.copy()

                return loss, gradient

            result = minimize(
                tracked_objective,
                initial_parameters(restart),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": MAX_ITERATIONS,
                    "gtol": GRADIENT_TOLERANCE,
                    "ftol": 1e-12,
                    "maxls": 40,
                },
            )

            require(
                tracked["theta"] is not None,
                "Optimizator nije dao konacan kandidat.",
            )

            if tracked["loss"] < best_loss:
                best_loss = tracked["loss"]
                best_theta = tracked["theta"].copy()
                best_status = str(result.message)

        loss, gradient = objective(
            best_theta, counts[slot], structure, regularization
        )
        models.append(best_theta)
        diagnostics.append({
            "loss": loss,
            "gradient": float(np.max(np.abs(gradient))),
            "status": best_status,
        })

    return models, diagnostics


def report_fit(label, diagnostics):
    print(label)

    for slot, diagnostic in enumerate(diagnostics, start=1):
        print(
            f"Registar {slot}, "
            f"Brier+L2={diagnostic['loss']:.9f}, "
            f"max|grad|={diagnostic['gradient']:.3e}"
        )

        if diagnostic["gradient"] > 10.0 * GRADIENT_TOLERANCE:
            print("Napomena: nije potvrdjen mali gradijent.")
            print(diagnostic["status"])


def base_log_probabilities(models, structure):
    logs = np.stack(
        [
            amplitude_logs(model_state(theta, structure))
            for theta in models
        ],
        axis=0,
    )

    require(
        np.allclose(
            logsumexp(logs, axis=1), 0.0,
            atol=2e-10, rtol=0.0,
        ),
        "Kvantne raspodele nisu normalizovane.",
    )
    return logs


def legal_logs(log_weights):
    logs = np.asarray(log_weights, dtype=np.float64).copy()
    require(logs.ndim == 2, "Ocekivana je matrica log masa.")
    require(
        not np.isnan(logs).any() and not np.isposinf(logs).any(),
        "Neispravne log mase.",
    )

    k, n = logs.shape
    require(1 <= k <= n, "Neispravne dimenzije.")

    for slot in range(k):
        logs[slot, :slot] = -np.inf
        logs[slot, n - k + slot + 1:] = -np.inf

    return logs


def strict_prefix_logsum(values):
    return np.concatenate([
        np.array([-np.inf]),
        np.logaddexp.accumulate(values)[:-1],
    ])


def joint_distribution(log_weights):
    logs = legal_logs(log_weights)
    k, n = logs.shape

    forward = np.full((k, n), -np.inf)
    maximum = np.full((k, n), -np.inf)
    parent = np.full((k, n), -1, dtype=np.int64)

    forward[0] = logs[0]
    maximum[0] = logs[0]

    for slot in range(1, k):
        forward[slot] = (
            logs[slot] + strict_prefix_logsum(forward[slot - 1])
        )

        running_index = 0

        for value in range(1, n):
            candidate = value - 1

            if (
                maximum[slot - 1, candidate]
                > maximum[slot - 1, running_index]
            ):
                running_index = candidate

            maximum[slot, value] = (
                logs[slot, value]
                + maximum[slot - 1, running_index]
            )
            parent[slot, value] = running_index

    log_z = float(logsumexp(forward[-1]))

    if not np.isfinite(log_z):
        raise DistributionError("Nema pozitivne legalne mase.")

    backward = np.full((k, n), -np.inf)
    backward[-1] = 0.0

    for slot in range(k - 2, -1, -1):
        next_values = logs[slot + 1] + backward[slot + 1]
        suffix = np.logaddexp.accumulate(next_values[::-1])[::-1]
        backward[slot] = np.concatenate([
            suffix[1:], np.array([-np.inf])
        ])

    marginals = np.exp(forward + backward - log_z)

    path = np.empty(k, dtype=np.int64)
    path[-1] = int(np.argmax(maximum[-1]))

    for slot in range(k - 1, 0, -1):
        path[slot - 1] = parent[slot, path[slot]]

    log_mode = float(logs[np.arange(k), path].sum())

    require(np.all(np.diff(path) > 0), "MAP nije strogo rastuci.")
    require(
        np.allclose(
            marginals.sum(axis=1), 1.0,
            atol=2e-9, rtol=0.0,
        ),
        "Uslovne marginale nisu normalizovane.",
    )
    require(
        log_mode <= log_z + 2e-9 and log_z <= 2e-9,
        "Neispravan normalizator ili MAP masa.",
    )

    return {
        "path": path,
        "logs": logs,
        "log_z": log_z,
        "marginals": marginals,
        "log_mode_probability": log_mode - log_z,
    }


def conditional_weights(base_logs, previous):
    return np.stack(
        [
            base_logs[slot, np.bitwise_xor(
                np.arange(N, dtype=np.int64),
                int(previous[slot]),
            )]
            for slot in range(K)
        ],
        axis=0,
    )


def validation_nll(models, structure, data, times):
    """
    Vektorizovan normalizator za sva validaciona kola.
    Racuna isti uslovni NLL kao joint_distribution.
    """
    times = np.asarray(list(times), dtype=np.int64)
    require(len(times) > 0, "Prazna validacija.")

    base_logs = base_log_probabilities(models, structure)
    previous = data[times - 1]
    targets = data[times]

    weights = np.stack(
        [
            base_logs[slot, np.bitwise_xor(
                previous[:, slot, None],
                np.arange(N, dtype=np.int64)[None, :],
            )]
            for slot in range(K)
        ],
        axis=0,
    )

    for slot in range(K):
        weights[slot, :, :slot] = -np.inf
        weights[slot, :, N - K + slot + 1:] = -np.inf

    forward = weights[0].copy()

    for slot in range(1, K):
        prefix = np.logaddexp.accumulate(forward, axis=1)
        strict_prefix = np.concatenate(
            [
                np.full((len(times), 1), -np.inf),
                prefix[:, :-1],
            ],
            axis=1,
        )
        forward = weights[slot] + strict_prefix

    log_z = logsumexp(forward, axis=1)

    if not np.isfinite(log_z).all():
        raise DistributionError("Nulta legalna masa u validaciji.")

    target_log_mass = np.zeros(len(times), dtype=np.float64)

    for slot in range(K):
        target_log_mass += base_logs[
            slot,
            np.bitwise_xor(previous[:, slot], targets[:, slot]),
        ]

    nll = log_z - target_log_mass

    require(
        not np.isnan(nll).any()
        and not np.isneginf(nll).any()
        and np.all(nll >= -2e-8),
        "Neispravan validacioni NLL.",
    )
    require(
        np.all(log_z <= 2e-9),
        "Validaciona legalna masa je veca od jedan.",
    )

    return float(np.mean(np.maximum(nll, 0.0)))


def evaluate(models, structure, data, times):
    base_logs = base_log_probabilities(models, structure)
    losses = []
    hits = []
    zero_targets = 0

    for time in times:
        joint = joint_distribution(
            conditional_weights(base_logs, data[time - 1])
        )
        target_log_mass = float(
            joint["logs"][np.arange(K), data[time]].sum()
        )

        if np.isneginf(target_log_mass):
            nll = np.inf
            zero_targets += 1
        else:
            nll = joint["log_z"] - target_log_mass
            require(
                np.isfinite(nll) and nll >= -2e-8,
                "Neispravan test NLL.",
            )
            nll = max(0.0, nll)

        losses.append(nll)
        hits.append(
            len(set(joint["path"].tolist()) & set(data[time].tolist()))
        )

    require(len(losses) > 0, "Prazan skup evaluacije.")

    return {
        "nll": float(np.mean(losses)),
        "hits": np.asarray(hits, dtype=np.int64),
        "zero_targets": zero_targets,
    }


def print_metrics(label, metrics):
    print(label)
    print(f"Srednji uslovni NLL: {metrics['nll']:.9f}")
    print(f"Srednji broj pogodaka: {metrics['hits'].mean():.6f}")
    print(f"Ciljevi sa nultom masom: {metrics['zero_targets']}")

    histogram = np.bincount(metrics["hits"], minlength=K + 1)

    for number in range(K + 1):
        print(f"{number} pogodaka: {int(histogram[number])}")


def self_checks(structures):
    require(
        np.allclose(
            HADAMARD.T @ HADAMARD,
            np.eye(DIM),
            atol=1e-12,
            rtol=0.0,
        ),
        "Hadamard nije unitaran.",
    )

    verify_simon(0)

    for period, structure in structures.items():
        require(
            verify_simon(period) == period,
            "Neuspesna rekonstrukcija perioda.",
        )

        theta = initial_parameters(0)

        for previous in (0, 38):
            forecast_qiskit(theta, structure, previous)

        probabilities = np.abs(
            model_state(theta, structure)
        ) ** 2

        require(
            np.allclose(
                probabilities,
                probabilities[np.bitwise_xor(INDICES, period)],
                atol=2e-12,
                rtol=0.0,
            ),
            "Prediktivni model ne cuva Simonove parove.",
        )

    counts = (
        1.0 + np.sin(np.arange(DIM, dtype=np.float64) + 0.3)
    ) ** 2

    for period in (1, 37, 63):
        structure = structures[period]
        theta = initial_parameters(0)
        _, analytic = objective(theta, counts, structure, 0.003)

        numeric = np.empty_like(theta)
        step = 1e-6

        for index in range(len(theta)):
            upper = theta.copy()
            lower = theta.copy()
            upper[index] += step
            lower[index] -= step

            numeric[index] = (
                objective(upper, counts, structure, 0.003)[0]
                - objective(lower, counts, structure, 0.003)[0]
            ) / (2.0 * step)

        require(
            np.allclose(
                analytic, numeric,
                atol=2e-6, rtol=2e-4,
            ),
            "Analiticki gradijent nije potvrdjen.",
        )

    toy = np.exp(
        0.5 * np.sin(np.arange(15, dtype=np.float64).reshape(3, 5))
    )
    toy /= toy.sum(axis=1, keepdims=True)
    toy_logs = np.log(toy)
    joint = joint_distribution(toy_logs)

    combinations = list(itertools.combinations(range(5), 3))
    scores = np.array([
        toy_logs[np.arange(3), combination].sum()
        for combination in combinations
    ])

    require(
        np.isclose(
            joint["log_z"], logsumexp(scores),
            atol=1e-12, rtol=0.0,
        ),
        "DP normalizator nije tacan.",
    )
    require(
        np.isclose(
            toy_logs[np.arange(3), joint["path"]].sum(),
            scores.max(),
            atol=1e-12, rtol=0.0,
        ),
        "DP nije dao globalni maksimum.",
    )

    brute_marginals = np.zeros_like(toy)

    for combination, mass in zip(
        combinations, np.exp(scores - logsumexp(scores))
    ):
        brute_marginals[np.arange(3), combination] += mass

    require(
        np.allclose(
            joint["marginals"], brute_marginals,
            atol=1e-12, rtol=0.0,
        ),
        "DP marginale nisu tacne.",
    )

    sole = np.full((3, 5), -np.inf)
    sole[np.arange(3), [0, 2, 4]] = 0.0
    sole_joint = joint_distribution(sole)

    require(
        np.array_equal(sole_joint["path"], [0, 2, 4])
        and np.isclose(sole_joint["log_z"], 0.0),
        "Greska kod jedine legalne kombinacije.",
    )

    impossible = np.full((3, 5), -np.inf)
    impossible[:, 0] = 0.0

    try:
        joint_distribution(impossible)
    except DistributionError:
        pass
    else:
        raise RuntimeError("Nulta legalna masa nije prijavljena.")

    fixture = np.array([
        [0, 4, 9, 13, 20, 30, 38],
        [1, 6, 10, 14, 22, 31, 37],
        [0, 5, 12, 16, 24, 33, 38],
    ], dtype=np.int64)

    structure = structures[37]
    models = [
        initial_parameters(0) + slot * 0.03
        for slot in range(K)
    ]

    vectorized = validation_nll(
        models, structure, fixture, range(1, len(fixture))
    )
    direct = evaluate(
        models, structure, fixture, range(1, len(fixture))
    )["nll"]

    require(
        np.isclose(vectorized, direct, atol=2e-10, rtol=0.0),
        "Vektorizovani i pojedinacni NLL se razlikuju.",
    )

    tiny = amplitude_logs(np.array([1e-180 + 0j]))[0]
    require(np.isfinite(tiny), "Izgubljena mala amplituda.")

    print("Ugradjene provere Simonovog kola i GF(2) su prosle.")
    print("Ugradjene provere amplituda, gradijenta i DP-a su prosle.")


def longest_consecutive(values):
    longest = 1
    current = 1

    for left, right in zip(values[:-1], values[1:]):
        current = current + 1 if right == left + 1 else 1
        longest = max(longest, current)

    return longest


def process_dataset(dataset, structures):
    path = dataset["path"]
    data = dataset["data"]

    transitions = len(data) - 1
    train_end = int(TRAIN_FRACTION * transitions)
    validation_end = int(VALIDATION_END_FRACTION * transitions)

    require(
        0 < train_end < validation_end < transitions,
        f"{path.name}: nedovoljno podataka za hronolosku podelu.",
    )

    print()
    print("=" * 72)
    print(path.name)
    print(f"Stvarni broj kola: {len(data)}")
    print(f"SHA256: {dataset['sha256']}")
    print(f"Trening prelazi: 1..{train_end}")
    print(f"Validacioni prelazi: {train_end + 1}..{validation_end}")
    print(f"Test prelazi: {validation_end + 1}..{transitions}")

    train_counts = transition_counts(data, train_end)
    screening = []
    first_regularization = REGULARIZATIONS[0]

    # Svih 63 nenultih perioda procenjuje se samo na treningu.
    for period, structure in structures.items():
        models, diagnostics = fit_models(
            train_counts, structure, first_regularization
        )
        score = float(np.mean([
            diagnostic["loss"] for diagnostic in diagnostics
        ]))

        screening.append({
            "period": period,
            "score": score,
            "models": models,
            "diagnostics": diagnostics,
        })

        print(
            f"Trening pregled b={period:06b}, "
            f"srednji Brier+L2={score:.9f}"
        )

    # Preselekcija je ogranicenje pretrage, bez uvida u validaciju.
    screening.sort(key=lambda item: (item["score"], item["period"]))
    shortlisted = screening[:PRESELECT_PERIODS]

    print()
    print(
        "Periodi izabrani iskljucivo treningom: "
        + ", ".join(f"{item['period']:06b}" for item in shortlisted)
    )

    best = None

    for item in shortlisted:
        period = item["period"]
        structure = structures[period]

        for regularization in REGULARIZATIONS:
            if regularization == first_regularization:
                models = item["models"]
                diagnostics = item["diagnostics"]
            else:
                models, diagnostics = fit_models(
                    train_counts, structure, regularization
                )

            label = (
                f"VALIDACIJA b={period:06b}, "
                f"L2={regularization}"
            )
            report_fit(label, diagnostics)

            try:
                score = validation_nll(
                    models,
                    structure,
                    data,
                    range(train_end + 1, validation_end + 1),
                )
            except DistributionError as exc:
                print(f"Kandidat odbijen: {exc}")
                continue

            print(f"Validacioni uslovni NLL: {score:.9f}")

            if not np.isfinite(score):
                print("Kandidat nema konacan validacioni NLL.")
                continue

            if best is None or score < best["score"]:
                best = {
                    "period": period,
                    "regularization": regularization,
                    "score": score,
                }

    require(
        best is not None,
        f"{path.name}: nema validnog modela.",
    )

    structure = structures[best["period"]]

    print()
    print(
        f"Izabrano b={best['period']:06b}, "
        f"L2={best['regularization']}"
    )
    print(
        f"Simonova GF(2) provera: "
        f"{verify_simon(best['period']):06b}"
    )

    # Test model vidi trening i validaciju, bez test ciljeva.
    test_models, diagnostics = fit_models(
        transition_counts(data, validation_end),
        structure,
        best["regularization"],
    )
    report_fit("TRENING ZA ODVOJENI TEST", diagnostics)

    metrics = evaluate(
        test_models,
        structure,
        data,
        range(validation_end + 1, len(data)),
    )
    print_metrics("ODVOJENI HRONOLOSKI TEST", metrics)

    # Zavrsno ucenje koristi svaki istorijski prelaz.
    final_models, diagnostics = fit_models(
        transition_counts(data, transitions),
        structure,
        best["regularization"],
    )
    report_fit("ZAVRSNO UCENJE NA CELOJ ISTORIJI", diagnostics)

    previous = data[-1]
    numerical_logs = conditional_weights(
        base_log_probabilities(final_models, structure),
        previous,
    )

    qiskit_logs = np.empty((K, N), dtype=np.float64)

    for slot in range(K):
        state = forecast_qiskit(
            final_models[slot],
            structure,
            int(previous[slot]),
        )
        qiskit_logs[slot] = amplitude_logs(state)[:N]

    require(
        np.allclose(
            np.exp(qiskit_logs),
            np.exp(numerical_logs),
            atol=2e-10,
            rtol=2e-9,
        ),
        "Zavrsna Qiskit raspodela se razlikuje od modela.",
    )

    joint = joint_distribution(qiskit_logs)
    combination = joint["path"] + 1

    require(
        len(combination) == K
        and len(set(combination.tolist())) == K
        and np.all((combination >= 1) & (combination <= N)),
        "Nelegalna NEXT kombinacija.",
    )

    print()
    print(f"Log legalne mase: {joint['log_z']:.9f}")
    print(
        f"Log uslovne mase MAP-a: "
        f"{joint['log_mode_probability']:.9f}"
    )

    base_logs = base_log_probabilities(final_models, structure)
    maximum_context_change = 0.0

    for slot in range(K):
        context_probabilities = np.exp(
            base_logs[slot, np.bitwise_xor(
                np.arange(N, dtype=np.int64)[:, None],
                np.arange(N, dtype=np.int64)[None, :],
            )]
        )
        maximum_context_change = max(
            maximum_context_change,
            float(np.max(np.abs(
                context_probabilities
                - context_probabilities[:, :1]
            ))),
        )

    print(
        f"Najveca promena raspodele kroz prethodne ulaze: "
        f"{maximum_context_change:.9e}"
    )

    if maximum_context_change < 1e-10:
        print("Upozorenje: gotovo nema zavisnosti od prethodnog ulaza.")

    if np.array_equal(combination, data[-1] + 1):
        print("Upozorenje: NEXT je isti kao poslednje kolo.")

    run = longest_consecutive(combination)

    if run >= 5:
        print(f"Upozorenje: niz od {run} uzastopnih brojeva.")

    span = int(combination[-1] - combination[0])
    historical_spans = data[:, -1] - data[:, 0]
    percentile = 100.0 * float(np.mean(historical_spans <= span))

    print(f"Raspon NEXT kombinacije: {span}")
    print(f"Udeo istorijskih raspona <= NEXT raspon: {percentile:.3f}%")

    if percentile <= 1.0:
        print("Upozorenje: izrazito grupisana NEXT kombinacija.")

    return {
        "name": path.name,
        "sha256": dataset["sha256"],
        "next": combination,
        "raw": np.exp(qiskit_logs),
        "marginals": joint["marginals"],
        "period": best["period"],
    }


def main():
    print("Quantum Simon")
    print(f"Qiskit: {qiskit.__version__}")
    print("Izvorna Simonova provera koristi dva registra po sest kubita.")
    print("GF(2) jednacine koriste tacnu podrsku izlaznog registra.")
    print("Nema slucajnog uzorkovanja merenja ili kombinacija.")
    print("Predikcija koristi sedam zasebnih sestokubitnih registara.")
    print("Pet RY uglova po registru uci amplitude Simonovih parova.")
    print("Period je hipoteza modela, a ne potvrdjen vremenski ciklus.")
    print("L-BFGS-B je klasicni optimizator kvantnih amplituda.")
    print("Svaki CSV se uci i obradjuje zasebno.")
    print()

    structures = {
        period: make_structure(period)
        for period in range(1, DIM)
    }

    self_checks(structures)

    results = []

    for path in FILES:
        dataset = load_csv(path)
        results.append(process_dataset(dataset, structures))

    print()
    print("=" * 72)
    print("POREDJENJE NEZAVISNIH REZULTATA")

    for left, right in itertools.combinations(results, 2):
        if left["sha256"] == right["sha256"]:
            print(
                f"Upozorenje: isti sadrzaj fajlova "
                f"{left['name']} i {right['name']}."
            )

        if np.array_equal(left["next"], right["next"]):
            print(
                f"Upozorenje: ista NEXT kombinacija za "
                f"{left['name']} i {right['name']}."
            )
            print(
                f"Izabrani periodi: "
                f"{left['period']:06b}, {right['period']:06b}"
            )
            print(
                "Najveca razlika sirovih raspodela: "
                f"{np.max(np.abs(left['raw'] - right['raw'])):.9e}"
            )
            print(
                "Najveca razlika uslovnih marginala: "
                f"{np.max(np.abs(left['marginals'] - right['marginals'])):.9e}"
            )
            print("Poklapanje zahteva pregled raspodela i testa.")

    print()
    print("=" * 72)

    for result in results:
        numbers = " ".join(
            str(int(value)) for value in result["next"]
        )
        print(f"{result['name']} NEXT: {numbers}")


if __name__ == "__main__":
    main()



"""
Quantum Simon
Qiskit: 1.4.4
Izvorna Simonova provera koristi dva registra po sest kubita.
GF(2) jednacine koriste tacnu podrsku izlaznog registra.
Nema slucajnog uzorkovanja merenja ili kombinacija.
Predikcija koristi sedam zasebnih sestokubitnih registara.
Pet RY uglova po registru uci amplitude Simonovih parova.
Period je hipoteza modela, a ne potvrdjen vremenski ciklus.
L-BFGS-B je klasicni optimizator kvantnih amplituda.
Svaki CSV se uci i obradjuje zasebno.

Ugradjene provere Simonovog kola i GF(2) su prosle.
Ugradjene provere amplituda, gradijenta i DP-a su prosle.

========================================================================
loto7_4716_k89.csv
Stvarni broj kola: 4716
SHA256: deb02ea741f500e3270402730ed89565f1c9b4618d53e5a08fea2b7c2c2af31a
Trening prelazi: 1..3300
Validacioni prelazi: 3301..4007
Test prelazi: 4008..4715
Trening pregled b=000001, srednji Brier+L2=0.970994070
Trening pregled b=000010, srednji Brier+L2=0.970979705
Trening pregled b=000011, srednji Brier+L2=0.970981573
Trening pregled b=000100, srednji Brier+L2=0.971106438
Trening pregled b=000101, srednji Brier+L2=0.971093598
Trening pregled b=000110, srednji Brier+L2=0.971116251
Trening pregled b=000111, srednji Brier+L2=0.971105671
Trening pregled b=001000, srednji Brier+L2=0.977281097
Trening pregled b=001001, srednji Brier+L2=0.977431615
Trening pregled b=001010, srednji Brier+L2=0.977445767
Trening pregled b=001011, srednji Brier+L2=0.977421893
Trening pregled b=001100, srednji Brier+L2=0.977404013
Trening pregled b=001101, srednji Brier+L2=0.977386301
Trening pregled b=001110, srednji Brier+L2=0.977400555
Trening pregled b=001111, srednji Brier+L2=0.977394808
Trening pregled b=010000, srednji Brier+L2=0.982382984
Trening pregled b=010001, srednji Brier+L2=0.985509658
Trening pregled b=010010, srednji Brier+L2=0.985531435
Trening pregled b=010011, srednji Brier+L2=0.985505971
Trening pregled b=010100, srednji Brier+L2=0.985417605
Trening pregled b=010101, srednji Brier+L2=0.985521370
Trening pregled b=010110, srednji Brier+L2=0.985556307
Trening pregled b=010111, srednji Brier+L2=0.985531453
Trening pregled b=011000, srednji Brier+L2=0.984388772
Trening pregled b=011001, srednji Brier+L2=0.982265425
Trening pregled b=011010, srednji Brier+L2=0.982267000
Trening pregled b=011011, srednji Brier+L2=0.982256363
Trening pregled b=011100, srednji Brier+L2=0.982221301
Trening pregled b=011101, srednji Brier+L2=0.982203181
Trening pregled b=011110, srednji Brier+L2=0.982208492
Trening pregled b=011111, srednji Brier+L2=0.982213628
Trening pregled b=100000, srednji Brier+L2=0.982218683
Trening pregled b=100001, srednji Brier+L2=0.982212054
Trening pregled b=100010, srednji Brier+L2=0.982203668
Trening pregled b=100011, srednji Brier+L2=0.982203713
Trening pregled b=100100, srednji Brier+L2=0.982214463
Trening pregled b=100101, srednji Brier+L2=0.982255780
Trening pregled b=100110, srednji Brier+L2=0.982271433
Trening pregled b=100111, srednji Brier+L2=0.982266633
Trening pregled b=101000, srednji Brier+L2=0.984353824
Trening pregled b=101001, srednji Brier+L2=0.985528297
Trening pregled b=101010, srednji Brier+L2=0.985557315
Trening pregled b=101011, srednji Brier+L2=0.985524267
Trening pregled b=101100, srednji Brier+L2=0.985422867
Trening pregled b=101101, srednji Brier+L2=0.985503047
Trening pregled b=101110, srednji Brier+L2=0.985530521
Trening pregled b=101111, srednji Brier+L2=0.985513027
Trening pregled b=110000, srednji Brier+L2=0.984402518
Trening pregled b=110001, srednji Brier+L2=0.977394733
Trening pregled b=110010, srednji Brier+L2=0.977397979
Trening pregled b=110011, srednji Brier+L2=0.977385675
Trening pregled b=110100, srednji Brier+L2=0.977397882
Trening pregled b=110101, srednji Brier+L2=0.977422670
Trening pregled b=110110, srednji Brier+L2=0.977448323
Trening pregled b=110111, srednji Brier+L2=0.977431437
Trening pregled b=111000, srednji Brier+L2=0.977403766
Trening pregled b=111001, srednji Brier+L2=0.971105894
Trening pregled b=111010, srednji Brier+L2=0.971105724
Trening pregled b=111011, srednji Brier+L2=0.971092374
Trening pregled b=111100, srednji Brier+L2=0.971116859
Trening pregled b=111101, srednji Brier+L2=0.970982778
Trening pregled b=111110, srednji Brier+L2=0.970990718
Trening pregled b=111111, srednji Brier+L2=0.970994376

Periodi izabrani iskljucivo treningom: 000010, 000011, 111101
VALIDACIJA b=000010, L2=0.0001
Registar 1, Brier+L2=0.965184359, max|grad|=1.343e-07
Registar 2, Brier+L2=0.980165416, max|grad|=4.188e-07
Registar 3, Brier+L2=0.983476683, max|grad|=1.193e-07
Registar 4, Brier+L2=0.984431693, max|grad|=1.380e-07
Registar 5, Brier+L2=0.977849744, max|grad|=6.828e-08
Registar 6, Brier+L2=0.963345911, max|grad|=6.487e-07
Registar 7, Brier+L2=0.942404129, max|grad|=3.031e-07
Validacioni uslovni NLL: 17.254427764
VALIDACIJA b=000010, L2=0.001
Registar 1, Brier+L2=0.972357236, max|grad|=3.550e-09
Registar 2, Brier+L2=0.989086548, max|grad|=7.203e-08
Registar 3, Brier+L2=0.992825307, max|grad|=2.781e-08
Registar 4, Brier+L2=0.993988226, max|grad|=1.574e-07
Registar 5, Brier+L2=0.986814674, max|grad|=1.060e-08
Registar 6, Brier+L2=0.970506071, max|grad|=4.736e-07
Registar 7, Brier+L2=0.948868049, max|grad|=3.601e-07
Validacioni uslovni NLL: 17.612487757
VALIDACIJA b=000011, L2=0.0001
Registar 1, Brier+L2=0.965218684, max|grad|=1.287e-07
Registar 2, Brier+L2=0.980163267, max|grad|=1.973e-08
Registar 3, Brier+L2=0.983469428, max|grad|=4.829e-08
Registar 4, Brier+L2=0.984425337, max|grad|=1.685e-07
Registar 5, Brier+L2=0.977844440, max|grad|=4.015e-08
Registar 6, Brier+L2=0.963339653, max|grad|=5.767e-07
Registar 7, Brier+L2=0.942410200, max|grad|=3.375e-07
Validacioni uslovni NLL: 17.262795593
VALIDACIJA b=000011, L2=0.001
Registar 1, Brier+L2=0.972573310, max|grad|=3.375e-07
Registar 2, Brier+L2=0.989078802, max|grad|=6.519e-08
Registar 3, Brier+L2=0.992917926, max|grad|=3.545e-08
Registar 4, Brier+L2=0.993926025, max|grad|=1.704e-07
Registar 5, Brier+L2=0.986747953, max|grad|=5.862e-09
Registar 6, Brier+L2=0.970472940, max|grad|=5.666e-07
Registar 7, Brier+L2=0.948865221, max|grad|=3.510e-07
Validacioni uslovni NLL: 17.613268943
VALIDACIJA b=111101, L2=0.0001
Registar 1, Brier+L2=0.965118013, max|grad|=7.425e-08
Registar 2, Brier+L2=0.980164321, max|grad|=1.158e-08
Registar 3, Brier+L2=0.983477910, max|grad|=1.658e-07
Registar 4, Brier+L2=0.984425668, max|grad|=3.038e-07
Registar 5, Brier+L2=0.977841695, max|grad|=2.882e-08
Registar 6, Brier+L2=0.963344835, max|grad|=1.591e-07
Registar 7, Brier+L2=0.942507007, max|grad|=2.310e-07
Validacioni uslovni NLL: 17.261522573
VALIDACIJA b=111101, L2=0.001
Registar 1, Brier+L2=0.972364878, max|grad|=3.149e-07
Registar 2, Brier+L2=0.989063232, max|grad|=7.244e-08
Registar 3, Brier+L2=0.992854667, max|grad|=1.524e-07
Registar 4, Brier+L2=0.993962184, max|grad|=2.039e-07
Registar 5, Brier+L2=0.986740355, max|grad|=6.089e-09
Registar 6, Brier+L2=0.970492365, max|grad|=7.059e-07
Registar 7, Brier+L2=0.949068631, max|grad|=6.238e-08
Validacioni uslovni NLL: 17.616544702

Izabrano b=000010, L2=0.0001
Simonova GF(2) provera: 000010
TRENING ZA ODVOJENI TEST
Registar 1, Brier+L2=0.965353247, max|grad|=1.995e-07
Registar 2, Brier+L2=0.980161375, max|grad|=4.855e-07
Registar 3, Brier+L2=0.983556410, max|grad|=1.208e-07
Registar 4, Brier+L2=0.984421130, max|grad|=1.432e-07
Registar 5, Brier+L2=0.977664417, max|grad|=7.180e-08
Registar 6, Brier+L2=0.962991895, max|grad|=4.345e-08
Registar 7, Brier+L2=0.942379077, max|grad|=5.210e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.296130690
Srednji broj pogodaka: 1.257062
Ciljevi sa nultom masom: 0
0 pogodaka: 160
1 pogodaka: 278
2 pogodaka: 205
3 pogodaka: 58
4 pogodaka: 7
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Registar 1, Brier+L2=0.965281822, max|grad|=1.417e-07
Registar 2, Brier+L2=0.980123727, max|grad|=7.692e-09
Registar 3, Brier+L2=0.983593031, max|grad|=8.331e-08
Registar 4, Brier+L2=0.984390089, max|grad|=1.479e-07
Registar 5, Brier+L2=0.977688605, max|grad|=8.906e-08
Registar 6, Brier+L2=0.962799730, max|grad|=9.342e-08
Registar 7, Brier+L2=0.942410011, max|grad|=2.881e-07

Log legalne mase: -8.146965314
Log uslovne mase MAP-a: -15.112902336
Najveca promena raspodele kroz prethodne ulaze: 6.612936364e-02
Raspon NEXT kombinacije: 30
Udeo istorijskih raspona <= NEXT raspon: 48.855%

========================================================================
loto7_4716_k89_loto_2980.csv
Stvarni broj kola: 2979
SHA256: f1cd81cd04660d50d9058675924da34346fc006ce8dabd84d52266c66afe17d1
Trening prelazi: 1..2084
Validacioni prelazi: 2085..2531
Test prelazi: 2532..2978
Trening pregled b=000001, srednji Brier+L2=0.970910634
Trening pregled b=000010, srednji Brier+L2=0.970913475
Trening pregled b=000011, srednji Brier+L2=0.970924483
Trening pregled b=000100, srednji Brier+L2=0.971045192
Trening pregled b=000101, srednji Brier+L2=0.971014384
Trening pregled b=000110, srednji Brier+L2=0.971031695
Trening pregled b=000111, srednji Brier+L2=0.971072167
Trening pregled b=001000, srednji Brier+L2=0.977199796
Trening pregled b=001001, srednji Brier+L2=0.977318258
Trening pregled b=001010, srednji Brier+L2=0.977318987
Trening pregled b=001011, srednji Brier+L2=0.977315651
Trening pregled b=001100, srednji Brier+L2=0.977253904
Trening pregled b=001101, srednji Brier+L2=0.977247156
Trening pregled b=001110, srednji Brier+L2=0.977261657
Trening pregled b=001111, srednji Brier+L2=0.977286039
Trening pregled b=010000, srednji Brier+L2=0.982361749
Trening pregled b=010001, srednji Brier+L2=0.985465800
Trening pregled b=010010, srednji Brier+L2=0.985461552
Trening pregled b=010011, srednji Brier+L2=0.985464146
Trening pregled b=010100, srednji Brier+L2=0.985332114
Trening pregled b=010101, srednji Brier+L2=0.985480946
Trening pregled b=010110, srednji Brier+L2=0.985486799
Trening pregled b=010111, srednji Brier+L2=0.985500089
Trening pregled b=011000, srednji Brier+L2=0.984354649
Trening pregled b=011001, srednji Brier+L2=0.982273806
Trening pregled b=011010, srednji Brier+L2=0.982260731
Trening pregled b=011011, srednji Brier+L2=0.982273116
Trening pregled b=011100, srednji Brier+L2=0.982237865
Trening pregled b=011101, srednji Brier+L2=0.982197370
Trening pregled b=011110, srednji Brier+L2=0.982198709
Trening pregled b=011111, srednji Brier+L2=0.982230599
Trening pregled b=100000, srednji Brier+L2=0.982231064
Trening pregled b=100001, srednji Brier+L2=0.982210315
Trening pregled b=100010, srednji Brier+L2=0.982198768
Trening pregled b=100011, srednji Brier+L2=0.982214100
Trening pregled b=100100, srednji Brier+L2=0.982233077
Trening pregled b=100101, srednji Brier+L2=0.982258776
Trening pregled b=100110, srednji Brier+L2=0.982258582
Trening pregled b=100111, srednji Brier+L2=0.982290056
Trening pregled b=101000, srednji Brier+L2=0.984334011
Trening pregled b=101001, srednji Brier+L2=0.985492191
Trening pregled b=101010, srednji Brier+L2=0.985488315
Trening pregled b=101011, srednji Brier+L2=0.985487950
Trening pregled b=101100, srednji Brier+L2=0.985331786
Trening pregled b=101101, srednji Brier+L2=0.985455305
Trening pregled b=101110, srednji Brier+L2=0.985460977
Trening pregled b=101111, srednji Brier+L2=0.985476040
Trening pregled b=110000, srednji Brier+L2=0.984362973
Trening pregled b=110001, srednji Brier+L2=0.977264751
Trening pregled b=110010, srednji Brier+L2=0.977266858
Trening pregled b=110011, srednji Brier+L2=0.977265662
Trening pregled b=110100, srednji Brier+L2=0.977261289
Trening pregled b=110101, srednji Brier+L2=0.977300342
Trening pregled b=110110, srednji Brier+L2=0.977312988
Trening pregled b=110111, srednji Brier+L2=0.977335716
Trening pregled b=111000, srednji Brier+L2=0.977280535
Trening pregled b=111001, srednji Brier+L2=0.971038295
Trening pregled b=111010, srednji Brier+L2=0.971039953
Trening pregled b=111011, srednji Brier+L2=0.971044597
Trening pregled b=111100, srednji Brier+L2=0.971054287
Trening pregled b=111101, srednji Brier+L2=0.970889995
Trening pregled b=111110, srednji Brier+L2=0.970908552
Trening pregled b=111111, srednji Brier+L2=0.970951669

Periodi izabrani iskljucivo treningom: 111101, 111110, 000001
VALIDACIJA b=111101, L2=0.0001
Registar 1, Brier+L2=0.965501137, max|grad|=2.504e-07
Registar 2, Brier+L2=0.979738637, max|grad|=4.092e-07
Registar 3, Brier+L2=0.983460874, max|grad|=4.271e-07
Registar 4, Brier+L2=0.984443968, max|grad|=1.902e-07
Registar 5, Brier+L2=0.977491364, max|grad|=7.388e-08
Registar 6, Brier+L2=0.963362223, max|grad|=1.195e-07
Registar 7, Brier+L2=0.942231764, max|grad|=5.150e-07
Validacioni uslovni NLL: 17.332539651
VALIDACIJA b=111101, L2=0.001
Registar 1, Brier+L2=0.972669541, max|grad|=2.023e-07
Registar 2, Brier+L2=0.988367345, max|grad|=1.159e-07
Registar 3, Brier+L2=0.992961166, max|grad|=1.682e-08
Registar 4, Brier+L2=0.993988446, max|grad|=2.575e-07
Registar 5, Brier+L2=0.986573394, max|grad|=6.551e-08
Registar 6, Brier+L2=0.970578267, max|grad|=3.644e-07
Registar 7, Brier+L2=0.948819084, max|grad|=2.569e-07
Validacioni uslovni NLL: 17.743437250
VALIDACIJA b=111110, L2=0.0001
Registar 1, Brier+L2=0.965581789, max|grad|=8.492e-07
Registar 2, Brier+L2=0.979780241, max|grad|=5.865e-09
Registar 3, Brier+L2=0.983471513, max|grad|=3.464e-07
Registar 4, Brier+L2=0.984443635, max|grad|=1.743e-07
Registar 5, Brier+L2=0.977518583, max|grad|=1.744e-07
Registar 6, Brier+L2=0.963314054, max|grad|=1.871e-07
Registar 7, Brier+L2=0.942250047, max|grad|=2.000e-07
Validacioni uslovni NLL: 17.312898458
VALIDACIJA b=111110, L2=0.001
Registar 1, Brier+L2=0.972794453, max|grad|=6.500e-07
Registar 2, Brier+L2=0.988512857, max|grad|=1.661e-07
Registar 3, Brier+L2=0.992908291, max|grad|=2.287e-08
Registar 4, Brier+L2=0.993920176, max|grad|=5.804e-07
Registar 5, Brier+L2=0.986501703, max|grad|=5.594e-08
Registar 6, Brier+L2=0.970434387, max|grad|=5.524e-08
Registar 7, Brier+L2=0.948715886, max|grad|=3.520e-07
Validacioni uslovni NLL: 17.694010989
VALIDACIJA b=000001, L2=0.0001
Registar 1, Brier+L2=0.965429302, max|grad|=2.259e-07
Registar 2, Brier+L2=0.979755036, max|grad|=3.888e-07
Registar 3, Brier+L2=0.983495681, max|grad|=8.315e-08
Registar 4, Brier+L2=0.984446238, max|grad|=9.709e-08
Registar 5, Brier+L2=0.977494491, max|grad|=1.391e-07
Registar 6, Brier+L2=0.963363930, max|grad|=2.042e-07
Registar 7, Brier+L2=0.942389763, max|grad|=5.927e-07
Validacioni uslovni NLL: 17.355140067
VALIDACIJA b=000001, L2=0.001
Registar 1, Brier+L2=0.972420068, max|grad|=7.307e-07
Registar 2, Brier+L2=0.988467441, max|grad|=1.450e-07
Registar 3, Brier+L2=0.993050778, max|grad|=1.416e-08
Registar 4, Brier+L2=0.994022860, max|grad|=1.847e-07
Registar 5, Brier+L2=0.986566008, max|grad|=3.978e-08
Registar 6, Brier+L2=0.970511778, max|grad|=8.447e-07
Registar 7, Brier+L2=0.949137178, max|grad|=2.434e-08
Validacioni uslovni NLL: 17.736247069

Izabrano b=111110, L2=0.0001
Simonova GF(2) provera: 111110
TRENING ZA ODVOJENI TEST
Registar 1, Brier+L2=0.965726505, max|grad|=1.402e-07
Registar 2, Brier+L2=0.980053132, max|grad|=1.320e-08
Registar 3, Brier+L2=0.983414432, max|grad|=4.171e-07
Registar 4, Brier+L2=0.984417575, max|grad|=2.052e-07
Registar 5, Brier+L2=0.977430356, max|grad|=1.003e-07
Registar 6, Brier+L2=0.963145761, max|grad|=3.607e-07
Registar 7, Brier+L2=0.942088383, max|grad|=8.919e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.501735060
Srednji broj pogodaka: 1.221477
Ciljevi sa nultom masom: 0
0 pogodaka: 106
1 pogodaka: 181
2 pogodaka: 122
3 pogodaka: 31
4 pogodaka: 7
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Registar 1, Brier+L2=0.966019818, max|grad|=1.057e-07
Registar 2, Brier+L2=0.980128591, max|grad|=1.385e-08
Registar 3, Brier+L2=0.983553656, max|grad|=4.302e-07
Registar 4, Brier+L2=0.984497197, max|grad|=3.172e-07
Registar 5, Brier+L2=0.977506530, max|grad|=6.148e-08
Registar 6, Brier+L2=0.963030083, max|grad|=4.042e-07
Registar 7, Brier+L2=0.942357877, max|grad|=3.060e-07

Log legalne mase: -8.648338543
Log uslovne mase MAP-a: -14.602438635
Najveca promena raspodele kroz prethodne ulaze: 6.566837820e-02
Raspon NEXT kombinacije: 20
Udeo istorijskih raspona <= NEXT raspon: 5.639%

========================================================================
loto7_4716_k89_loto_plus_1736.csv
Stvarni broj kola: 1735
SHA256: 56225ed0f0be7e2d3e30779804d8fa45e96ecfb9d14441eb2a731dd7fff38ae6
Trening prelazi: 1..1213
Validacioni prelazi: 1214..1473
Test prelazi: 1474..1734
Trening pregled b=000001, srednji Brier+L2=0.970927736
Trening pregled b=000010, srednji Brier+L2=0.970971174
Trening pregled b=000011, srednji Brier+L2=0.970871546
Trening pregled b=000100, srednji Brier+L2=0.971222923
Trening pregled b=000101, srednji Brier+L2=0.971200717
Trening pregled b=000110, srednji Brier+L2=0.971242223
Trening pregled b=000111, srednji Brier+L2=0.971234981
Trening pregled b=001000, srednji Brier+L2=0.977123919
Trening pregled b=001001, srednji Brier+L2=0.977439897
Trening pregled b=001010, srednji Brier+L2=0.977500236
Trening pregled b=001011, srednji Brier+L2=0.977421375
Trening pregled b=001100, srednji Brier+L2=0.977321986
Trening pregled b=001101, srednji Brier+L2=0.977289967
Trening pregled b=001110, srednji Brier+L2=0.977343777
Trening pregled b=001111, srednji Brier+L2=0.977313093
Trening pregled b=010000, srednji Brier+L2=0.982222760
Trening pregled b=010001, srednji Brier+L2=0.985396640
Trening pregled b=010010, srednji Brier+L2=0.985420169
Trening pregled b=010011, srednji Brier+L2=0.985382048
Trening pregled b=010100, srednji Brier+L2=0.985059135
Trening pregled b=010101, srednji Brier+L2=0.985460767
Trening pregled b=010110, srednji Brier+L2=0.985491720
Trening pregled b=010111, srednji Brier+L2=0.985472291
Trening pregled b=011000, srednji Brier+L2=0.984327448
Trening pregled b=011001, srednji Brier+L2=0.982327113
Trening pregled b=011010, srednji Brier+L2=0.982339066
Trening pregled b=011011, srednji Brier+L2=0.982278144
Trening pregled b=011100, srednji Brier+L2=0.982175378
Trening pregled b=011101, srednji Brier+L2=0.982146152
Trening pregled b=011110, srednji Brier+L2=0.982147388
Trening pregled b=011111, srednji Brier+L2=0.982154218
Trening pregled b=100000, srednji Brier+L2=0.982158190
Trening pregled b=100001, srednji Brier+L2=0.982162025
Trening pregled b=100010, srednji Brier+L2=0.982171312
Trening pregled b=100011, srednji Brier+L2=0.982120130
Trening pregled b=100100, srednji Brier+L2=0.982153878
Trening pregled b=100101, srednji Brier+L2=0.982302383
Trening pregled b=100110, srednji Brier+L2=0.982313495
Trening pregled b=100111, srednji Brier+L2=0.982312748
Trening pregled b=101000, srednji Brier+L2=0.984241676
Trening pregled b=101001, srednji Brier+L2=0.985471556
Trening pregled b=101010, srednji Brier+L2=0.985505634
Trening pregled b=101011, srednji Brier+L2=0.985454574
Trening pregled b=101100, srednji Brier+L2=0.985065956
Trening pregled b=101101, srednji Brier+L2=0.985388795
Trening pregled b=101110, srednji Brier+L2=0.985407915
Trening pregled b=101111, srednji Brier+L2=0.985399859
Trening pregled b=110000, srednji Brier+L2=0.984313099
Trening pregled b=110001, srednji Brier+L2=0.977299763
Trening pregled b=110010, srednji Brier+L2=0.977354676
Trening pregled b=110011, srednji Brier+L2=0.977287738
Trening pregled b=110100, srednji Brier+L2=0.977317427
Trening pregled b=110101, srednji Brier+L2=0.977425120
Trening pregled b=110110, srednji Brier+L2=0.977484985
Trening pregled b=110111, srednji Brier+L2=0.977447602
Trening pregled b=111000, srednji Brier+L2=0.977321345
Trening pregled b=111001, srednji Brier+L2=0.971245765
Trening pregled b=111010, srednji Brier+L2=0.971281836
Trening pregled b=111011, srednji Brier+L2=0.971173051
Trening pregled b=111100, srednji Brier+L2=0.971263177
Trening pregled b=111101, srednji Brier+L2=0.970899579
Trening pregled b=111110, srednji Brier+L2=0.970934886
Trening pregled b=111111, srednji Brier+L2=0.970931437

Periodi izabrani iskljucivo treningom: 000011, 111101, 000001
VALIDACIJA b=000011, L2=0.0001
Registar 1, Brier+L2=0.963941993, max|grad|=5.348e-07
Registar 2, Brier+L2=0.980006129, max|grad|=3.726e-08
Registar 3, Brier+L2=0.983822161, max|grad|=1.331e-07
Registar 4, Brier+L2=0.984345558, max|grad|=6.411e-08
Registar 5, Brier+L2=0.978487714, max|grad|=9.910e-08
Registar 6, Brier+L2=0.963053615, max|grad|=4.090e-08
Registar 7, Brier+L2=0.942443648, max|grad|=1.683e-07
Validacioni uslovni NLL: 17.168608544
VALIDACIJA b=000011, L2=0.001
Registar 1, Brier+L2=0.970725003, max|grad|=7.860e-09
Registar 2, Brier+L2=0.988873711, max|grad|=6.463e-08
Registar 3, Brier+L2=0.993064297, max|grad|=6.517e-07
Registar 4, Brier+L2=0.993686524, max|grad|=5.886e-07
Registar 5, Brier+L2=0.987676273, max|grad|=3.127e-08
Registar 6, Brier+L2=0.970023683, max|grad|=2.072e-07
Registar 7, Brier+L2=0.948614242, max|grad|=2.210e-07
Validacioni uslovni NLL: 17.482152990
VALIDACIJA b=111101, L2=0.0001
Registar 1, Brier+L2=0.963722093, max|grad|=1.521e-07
Registar 2, Brier+L2=0.980006286, max|grad|=1.136e-08
Registar 3, Brier+L2=0.983919962, max|grad|=2.565e-07
Registar 4, Brier+L2=0.984359775, max|grad|=5.034e-08
Registar 5, Brier+L2=0.978520981, max|grad|=6.175e-08
Registar 6, Brier+L2=0.963078319, max|grad|=1.325e-07
Registar 7, Brier+L2=0.942689639, max|grad|=7.659e-08
Validacioni uslovni NLL: 17.169214373
VALIDACIJA b=111101, L2=0.001
Registar 1, Brier+L2=0.970521450, max|grad|=1.438e-07
Registar 2, Brier+L2=0.988898856, max|grad|=5.822e-08
Registar 3, Brier+L2=0.993426058, max|grad|=2.545e-07
Registar 4, Brier+L2=0.993765770, max|grad|=2.619e-07
Registar 5, Brier+L2=0.987485150, max|grad|=2.114e-07
Registar 6, Brier+L2=0.970126148, max|grad|=1.077e-07
Registar 7, Brier+L2=0.949086074, max|grad|=2.416e-08
Validacioni uslovni NLL: 17.465223636
VALIDACIJA b=000001, L2=0.0001
Registar 1, Brier+L2=0.963628457, max|grad|=2.958e-07
Registar 2, Brier+L2=0.980005538, max|grad|=7.370e-08
Registar 3, Brier+L2=0.983920298, max|grad|=3.316e-08
Registar 4, Brier+L2=0.984325512, max|grad|=1.394e-07
Registar 5, Brier+L2=0.978527919, max|grad|=7.998e-08
Registar 6, Brier+L2=0.962951444, max|grad|=2.345e-07
Registar 7, Brier+L2=0.943134986, max|grad|=2.337e-07
Validacioni uslovni NLL: 17.174297042
VALIDACIJA b=000001, L2=0.001
Registar 1, Brier+L2=0.970216221, max|grad|=8.348e-08
Registar 2, Brier+L2=0.988897796, max|grad|=1.990e-08
Registar 3, Brier+L2=0.993380040, max|grad|=4.949e-07
Registar 4, Brier+L2=0.993527031, max|grad|=4.291e-07
Registar 5, Brier+L2=0.987508115, max|grad|=2.085e-08
Registar 6, Brier+L2=0.969880250, max|grad|=2.860e-07
Registar 7, Brier+L2=0.949687755, max|grad|=1.871e-08
Validacioni uslovni NLL: 17.484749419

Izabrano b=000011, L2=0.0001
Simonova GF(2) provera: 000011
TRENING ZA ODVOJENI TEST
Registar 1, Brier+L2=0.964120146, max|grad|=3.992e-08
Registar 2, Brier+L2=0.979799638, max|grad|=6.031e-07
Registar 3, Brier+L2=0.983632480, max|grad|=1.272e-07
Registar 4, Brier+L2=0.984318940, max|grad|=3.892e-08
Registar 5, Brier+L2=0.978109852, max|grad|=1.648e-08
Registar 6, Brier+L2=0.962891575, max|grad|=1.048e-07
Registar 7, Brier+L2=0.942093081, max|grad|=2.804e-07
ODVOJENI HRONOLOSKI TEST
Srednji uslovni NLL: 17.122566851
Srednji broj pogodaka: 1.268199
Ciljevi sa nultom masom: 0
0 pogodaka: 50
1 pogodaka: 116
2 pogodaka: 72
3 pogodaka: 21
4 pogodaka: 2
5 pogodaka: 0
6 pogodaka: 0
7 pogodaka: 0
ZAVRSNO UCENJE NA CELOJ ISTORIJI
Registar 1, Brier+L2=0.963668970, max|grad|=1.787e-07
Registar 2, Brier+L2=0.979611292, max|grad|=2.925e-07
Registar 3, Brier+L2=0.983671581, max|grad|=1.446e-07
Registar 4, Brier+L2=0.984370305, max|grad|=1.370e-07
Registar 5, Brier+L2=0.978029817, max|grad|=3.659e-08
Registar 6, Brier+L2=0.962228361, max|grad|=5.014e-07
Registar 7, Brier+L2=0.942278075, max|grad|=3.265e-07

Log legalne mase: -8.771910994
Log uslovne mase MAP-a: -14.158883160
Najveca promena raspodele kroz prethodne ulaze: 7.222963840e-02
Raspon NEXT kombinacije: 30
Udeo istorijskih raspona <= NEXT raspon: 48.242%

========================================================================
POREDJENJE NEZAVISNIH REZULTATA

========================================================================

loto7_4716_k89.csv NEXT: 6 x 17 y 26 z 36

loto7_4716_k89_loto_2980.csv NEXT: 12 x 17 y 28 z 32

loto7_4716_k89_loto_plus_1736.csv NEXT: 9 x 17 y 23 z 39
"""



"""
Q6-Q10	                            Osnovna ideja	                                                        Primena
Single Qubit Gates	                Rotacije, relativne faze i promene merne baze.	                        Učenje parametara jednokubitnih kapija. 
Multiple Qubits and Entanglement	Tenzorski proizvod, kontrolisane kapije i zapletenost.	                Učenje zapletenih stanja unutar šestokubitnih registara. 
Shor	                            Modularna eksponencijacija, procena faze i nalaženje reda.	            Posebna matematička veza sa istorijom kola. 
Bernstein-Vazirani	                Oporavak skrivenog niza iz linearnog binarnog orakla.	                Orakl izveden iz podataka koji zadovoljava taj uslov. 
Simon	                            Otkrivanje skrivene XOR simetrije funkcije.	                            Funkcija sa odgovarajućom strukturom preslikavanja. 


 6. Quantum_Single_Qubit_Gates.py
 7. Quantum_Entanglement.py
 8. Shor
 9. Bernstein-Vazirani
10. Simon

Za poslednja tri prvo treba definisati orakl iz istorije i proveriti uslove izvornog algoritma. 



--------
6. 

Quantum_Single_Qubit_Gates

Jednokubitna rotacija i fazna interferencija. 
Svaki šestokubitni registar ima proizvoljno stanje. 
Parametri kapija iz istorijskih prelaza kroz regresiju raspodele.



-------------
7. 

Quantum_Entanglement

Kontrolisane kapije i zapletenost. 
Između slojeva naučenih rotacija koriste se kontrolisane operacije. 
Provera redukovanih stanja pokazuje da li kolo zaista stvara zapletenost.

Rotacije zavisne od istorije pripremaju stanje, zatim CNOT lanac povezuje šest kubita, a završne rotacije menjaju mernu bazu. 
Provera zapletenosti preko čistoće redukovanih stanja i poređenje amplituda sa Qiskitom pre obuke.

Koristi naučene rotacije i CNOT lanac. 
Zapletenost se proverava unutar svakog šestokubitnog registra preko redukovanih matrica gustine. 
Redosled bitova prati Qiskit 1.4.

- smer i inverziju CNOT permutacija
- kompleksni gradijent kroz RY, RZ i CNOT
- faktor gradijenta Brierovog gubitka
- Bell i GHZ provere
- čistoća redukovanih stanja i Qiskit redosled bitova
- logaritamska normalizacija, globalni maksimum i hronološka podela 


Svaki kubit koristi niz RY → RZ → RY, koji pretvara relativnu fazu u promenu izlaznih verovatnoća. 
Prozor istorije bira se validacijom. 
Konačna kombinacija dobija se kao globalni maksimum raspodele nad svim legalnim izborima sedam brojeva.

Koristim sedam šestokubitnih registara i niz RY → RZ → RY na svakom kubitu. 
Prozore od jednog i tri prethodna kola biram validacijom. 
Optimizator uči parametre kapija iz cele trening istorije.


- izvodi verovatnoća po sva tri ugla RY → RZ → RY
- prefiksni i sufiksni proizvodi u gradijentu
- dimenzije parametara za oba prozora istorije
- redosled bitova prema Qiskitu
- logaritamska normalizacija i globalni izbor maksimuma
- granice treninga, validacije, testa i završnog učenja.



-------------
8. 

Shorov deo modularna eksponencijacija, QPE, inverzna QFT, procena faze i verizni ralomci za nalaženje reda. 

Orakl je \(f_a(x)=a^x\bmod 41\), uz izbor baze a hronološkom validacijom. 
Modul 41 omogućava modularni inverz za svaki broj od 1 do 39. 
QPE odredjuje red, a dodatni model uči amplitude koraka po modularnoj orbiti. 
Superpozicija periodičnih sopstvenih stanja modularnog operatora, čiju dimenziju određuje red dobijen QPE postupkom. 
Provera Shorove faktorizacije broja 15. 
Nalaženje reda koristiće 12 pomoćnih kubita i šest radnih kubita, a poređenje sa Qiskitom manju QPE konstrukciju.

Prediktivna dopuna uči superpoziciju periodičnih sopstvenih stanja modularnog operatora. 
Koristi modul 41, poslednje kolo kao početak orbite i celu istoriju prelaza za obuku. 
Bazu i broj koeficijenata bira validacijom.

QPE simulira 12 pomoćnih i šest radnih kubita. 
Manja QPE konstrukcija poredi se sa Qiskitom. 
Prediktivni deo koristi sedam šestokubitnih registara.



-------------
9.

Quantum_Bernstein_Vazirani 

Orakl, fazni povratni uticaj i Hadamard transformacije. 
Hadamard transformacije, linearni BV orakl i fazni povratni uticaj. 
Za Loto učim dodatnu fazu orakla iz istorijskih prelaza. 
Prethodno kolo je poznati ulaz. Buduća kombinacija se ne ugrađuje u orakl.

Koristim sedam registara sa po šest kubita i po jedan pomoćni kubit. 
Registri se simuliraju zasebno. 



----------------
10. 

Quantum_Simon

Sačuvaću orakl sa XOR periodom i rešavanje jednačina nad GF(2). 

Amplitude na parovima stanja \(x\) i \(x \oplus b\). 
Kandidati za XOR period biraju se iz treninga, a konačni model hronološkom validacijom. 
Simonovo kolo proverava period konstruisanog orakla. 

Dvostruki registar, orakl sa XOR periodom, Hadamard transformacije i rešavanje jednačina nad GF(2). 
Amplitude na Simonovim parovima stanja. 
Kandidati za period procenjuju se na treningu, a konačni model bira hronološkom validacijom.

Simonova provera koristi 12 kubita. 
Predikcija koristi sedam zasebnih registara sa po šest kubita. 
"""
