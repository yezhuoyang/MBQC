# test.py
# Test the correctness of the MBQC compiler by comparing
# logical Pauli frames between:
#   (1) A Clifford-level interpreter of the gadgets, and
#   (2) The compiled QMeas program run by the MBQC simulator.

from simulator import Simulator
from clifford_simulator import CliffordSimulator
from clifford import (
    QRegDecl as CliffordQRegDecl,
    CliffordInstruction,
    CNOT,
    SingleQubitGate,
    CliffordQubitRef,
)
from compiler import MBQCCompiler
from language import (
    QRegDecl as QMeasQRegDecl,
    CRegDecl as QMeasCRegDecl,
    MeasurementInstruction,
    FrameInstruction,
    IfInstruction,
    QubitRef,
    CBitRef,
)

import random


# ---------------------------------------------------------------------------
# Helper: outcome stream
# ---------------------------------------------------------------------------

class OutcomeStream:
    """
    Deterministic stream of ±1 outcomes.

    Used by both the Clifford-frame interpreter and the MBQC simulator so that
    they see the exact same measurement results.
    """

    def __init__(self, values):
        self.values = list(values)
        self.pos = 0

    @classmethod
    def random(cls, length, rng=None):
        if rng is None:
            rng = random
        vals = [rng.choice([-1, +1]) for _ in range(length)]
        return cls(vals)

    def next(self) -> int:
        if self.pos >= len(self.values):
            raise IndexError("OutcomeStream exhausted")
        v = self.values[self.pos]
        self.pos += 1
        return v

    def reset(self):
        self.pos = 0


# ---------------------------------------------------------------------------
# Clifford-level Pauli-frame interpreter for gadgets
# ---------------------------------------------------------------------------

class CliffordFrameSimulator:
    """
    A *logical* Pauli-frame simulator for Clifford circuits, using the same
    gadget semantics as the MBQC compiler.

    - It does NOT track a full stabilizer state; it only tracks a Pauli frame
      on logical qubits plus logical-to-physical mapping (matching MBQCCompiler).
    - For each gate, it consumes the same number of measurement outcomes
      that the MBQC gadget uses, from a shared OutcomeStream.
    - It updates the frame using the same classical rules as the QMeas gadgets.
    """

    def __init__(self, clifford_prog, outcome_stream: OutcomeStream):
        self.clifford_prog = clifford_prog
        self.outcomes = outcome_stream

        # Logical mapping: (reg_name, logical_idx) -> physical index
        self.base_reg_name = None
        self.base_reg_size = 0
        self.logical_to_physical = {}  # same shape as in MBQCCompiler

        # Pauli frame: physical_index -> 'I','X','Y','Z'
        self.frame = {}
        # Next ancilla index (physical)
        self.next_ancilla = 0

    # --- Pauli-frame helpers ------------------------------------------------

    @staticmethod
    def _compose_pauli(p_old: str, p_new: str) -> str:
        """
        Compose Pauli operators ignoring phase:

            I * A = A; A * I = A
            X * X = I; Y * Y = I; Z * Z = I
            X * Y = Z; Y * X = Z
            X * Z = Y; Z * X = Y
            Y * Z = X; Z * Y = X
        """
        if p_old == "I":
            return p_new
        if p_new == "I":
            return p_old
        if p_old == p_new:
            return "I"

        pair = {p_old, p_new}
        if pair == {"X", "Y"}:
            return "Z"
        if pair == {"X", "Z"}:
            return "Y"
        if pair == {"Y", "Z"}:
            return "X"
        # Should not reach here for {I,X,Y,Z}
        return "I"

    def _frame_update(self, phys_idx: int, pauli: str):
        old = self.frame.get(phys_idx, "I")
        self.frame[phys_idx] = self._compose_pauli(old, pauli)

    # --- Initialization -----------------------------------------------------

    def _init_mapping_and_frame(self, qreg: CliffordQRegDecl):
        self.base_reg_name = qreg.reg_name
        self.base_reg_size = qreg.size
        self.next_ancilla = qreg.size

        self.logical_to_physical = {
            (self.base_reg_name, i): i for i in range(self.base_reg_size)
        }
        self.frame = {i: "I" for i in range(self.base_reg_size)}

    # --- Main API -----------------------------------------------------------

    def run(self):
        for instr in self.clifford_prog:
            if isinstance(instr, CliffordQRegDecl):
                self._init_mapping_and_frame(instr)
            elif isinstance(instr, CliffordInstruction):
                self._handle_gate(instr)
            else:
                # ignore anything else for now
                pass

    # --- Gate handling ------------------------------------------------------

    def _lookup_phys(self, qref: CliffordQubitRef) -> int:
        key = (qref.reg_name, qref.index)
        return self.logical_to_physical[key]

    def _alloc_ancilla(self) -> int:
        idx = self.next_ancilla
        self.next_ancilla += 1
        # frame on fresh ancilla is identity
        self.frame[idx] = "I"
        return idx

    def _handle_gate(self, instr: CliffordInstruction):
        name = instr.name

        if name == "H":
            self._handle_H(instr)
        elif name == "S":
            self._handle_S(instr)
        elif name == "CNOT":
            self._handle_CNOT(instr)
        elif name in {"X", "Y", "Z"}:
            self._handle_Pauli(instr)
        else:
            raise ValueError(f"Unsupported Clifford gate in frame simulator: {name!r}")

    def _handle_H(self, instr: CliffordInstruction):
        """
        Gadget:

            r1 := MZZ(q, a);
            r2 := MX(q);
            if r1 = -1 then frame_Z(a);
            if r2 = -1 then frame_X(a);
            discard q.

        Logical qubit moves from old location of q to ancilla a.
        """
        if len(instr.qubits) != 1:
            raise ValueError("H must have 1 qubit")

        q_log = instr.qubits[0]
        _phys_q = self._lookup_phys(q_log)

        # Allocate ancilla
        a = self._alloc_ancilla()

        # Consume two measurement outcomes
        r1 = self.outcomes.next()
        r2 = self.outcomes.next()

        # Apply same classical frame rules
        if r1 == -1:
            self._frame_update(a, "Z")
        if r2 == -1:
            self._frame_update(a, "X")

        # Move logical location
        key = (q_log.reg_name, q_log.index)
        self.logical_to_physical[key] = a

    def _handle_S(self, instr: CliffordInstruction):
        """
        Gadget:

            r1 := MYZ(q, a);
            r2 := MX(q);
            if r1 = -1 then frame_Z(a);
            if r2 = -1 then frame_X(a);
            discard q.

        Logical qubit moves from q to a.
        """
        if len(instr.qubits) != 1:
            raise ValueError("S must have 1 qubit")

        q_log = instr.qubits[0]
        _phys_q = self._lookup_phys(q_log)

        a = self._alloc_ancilla()

        r1 = self.outcomes.next()
        r2 = self.outcomes.next()

        if r1 == -1:
            self._frame_update(a, "Z")
        if r2 == -1:
            self._frame_update(a, "X")

        key = (q_log.reg_name, q_log.index)
        self.logical_to_physical[key] = a

    def _handle_CNOT(self, instr: CliffordInstruction):
        """
        Gadget:

            r1 := MZZ(c, a1);
            r2 := MXX(a1, t);
            r3 := MZZ(a1, a2);
            r4 := MX(a1);
            if r1 = -1 then frame_Z(t);
            if r2 = -1 then frame_X(c);
            if r3 = -1 then frame_Z(c);
            if r4 = -1 then frame_X(t);
            discard a1,a2.

        Logical qubits remain at c and t.
        """
        if len(instr.qubits) != 2:
            raise ValueError("CNOT must have 2 qubits")

        c_log = instr.qubits[0]
        t_log = instr.qubits[1]

        phys_c = self._lookup_phys(c_log)
        phys_t = self._lookup_phys(t_log)

        # ancillas
        _a1 = self._alloc_ancilla()
        _a2 = self._alloc_ancilla()

        # outcomes
        r1 = self.outcomes.next()
        r2 = self.outcomes.next()
        r3 = self.outcomes.next()
        r4 = self.outcomes.next()

        # Classical corrections
        if r1 == -1:
            self._frame_update(phys_t, "Z")
        if r2 == -1:
            self._frame_update(phys_c, "X")
        if r3 == -1:
            self._frame_update(phys_c, "Z")
        if r4 == -1:
            self._frame_update(phys_t, "X")

        # logical locations unchanged

    def _handle_Pauli(self, instr: CliffordInstruction):
        """
        Pauli X/Y/Z compiled as pure frame updates:

            X(q) -> frame_X(q)
            Y(q) -> frame_Y(q)
            Z(q) -> frame_Z(q)
        """
        if len(instr.qubits) != 1:
            raise ValueError("Pauli must have 1 qubit")

        q_log = instr.qubits[0]
        phys_q = self._lookup_phys(q_log)

        name = instr.name
        if name == "X":
            p = "X"
        elif name == "Y":
            p = "Y"
        elif name == "Z":
            p = "Z"
        else:
            raise ValueError(f"Unknown Pauli {name!r}")
        self._frame_update(phys_q, p)

    # --- Final logical frame ------------------------------------------------

    def logical_frame(self):
        """
        Return a dict:
            (reg_name, logical_idx) -> Pauli ('I','X','Y','Z')
        obtained by reading the frame on the physical location of each logical qubit.
        """
        result = {}
        for (rname, lidx), pidx in self.logical_to_physical.items():
            result[(rname, lidx)] = self.frame.get(pidx, "I")
        return result


# ---------------------------------------------------------------------------
# Deterministic MBQC simulator (driven by OutcomeStream)
# ---------------------------------------------------------------------------

class DeterministicMBQCSimulator(Simulator):
    """
    Subclass of Simulator that uses a given OutcomeStream instead of random
    coin flips for measurement outcomes. It also lets us inspect the Pauli
    frame at the end.
    """

    def __init__(self, outcome_stream: OutcomeStream):
        super().__init__()
        self.outcomes = outcome_stream

    def _simulate_measurement(self, meas: MeasurementInstruction):
        """
        Override the default coin flip: pull next outcome from OutcomeStream.
        """
        cbit: CBitRef = meas.classical_bit
        c_index = cbit.index

        outcome = self.outcomes.next()
        self.classical_store[c_index] = outcome

        print(f"[MBQC] Measure {meas.name} on "
              f"{', '.join(str(q) for q in meas.qubits)} -> {outcome}")


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def assert_equal(a, b, msg=""):
    if a != b:
        raise AssertionError(f"{msg} (expected {b}, got {a})")


def print_program(prog, title=None):
    if title:
        print(f"=== {title} ===")
    for instr in prog:
        print(instr)
    print()


# ---------------------------------------------------------------------------
# Core test: one Clifford circuit vs compiled MBQC
# ---------------------------------------------------------------------------

def run_equivalence_test(clifford_prog, name=""):
    """
    For a given Clifford program:
      1. Count how many measurement outcomes the compiler will use.
      2. Create a random OutcomeStream of that length.
      3. Run CliffordFrameSimulator using these outcomes.
      4. Compile to QMeas and run DeterministicMBQCSimulator using the same outcomes.
      5. Compare final logical Pauli frames.
    """
    print(f"\n==============================")
    print(f"Test: {name}")
    print(f"==============================")

    # First compile to know measurement count.
    compiler = MBQCCompiler(clifford_prog)
    compiler.compile()
    qmeas_prog = compiler.get_compiled_program()

    # Count how many MeasurementInstructions we have:
    meas_count = sum(1 for instr in qmeas_prog if isinstance(instr, MeasurementInstruction))

    # Build OutcomeStream
    rng = random.Random(12345)  # fixed seed for reproducibility
    outcomes = OutcomeStream.random(meas_count, rng=rng)

    # 1) Clifford-level frame simulation
    cf = CliffordFrameSimulator(clifford_prog, OutcomeStream(outcomes.values))
    cf.run()
    cliff_frame = cf.logical_frame()
    print("Clifford logical frame:", cliff_frame)

    # 2) MBQC simulation with same outcomes
    mbqc_sim = DeterministicMBQCSimulator(OutcomeStream(outcomes.values))
    mbqc_sim.load_program(qmeas_prog)
    mbqc_sim.run()

    # Use compiler's logical_to_physical to project physical frame to logical
    logical_frame_mbqc = {}
    for (rname, lidx), qref in compiler.logical_to_physical.items():
        phys_idx = qref.index
        p = mbqc_sim.pauli_frame.get(phys_idx, "I")
        logical_frame_mbqc[(rname, lidx)] = p

    print("MBQC logical frame:", logical_frame_mbqc)

    # Compare
    assert_equal(cliff_frame, logical_frame_mbqc, "Logical frames differ")
    print("Result: OK (logical frames match)")


# ---------------------------------------------------------------------------
# Simple / base tests
# ---------------------------------------------------------------------------

def test_H():
    prog = [
        CliffordQRegDecl("q", 1),
        SingleQubitGate("H", CliffordQubitRef("q", 0)),
    ]
    run_equivalence_test(prog, "H on q[0]")


def test_S():
    prog = [
        CliffordQRegDecl("q", 1),
        SingleQubitGate("S", CliffordQubitRef("q", 0)),
    ]
    run_equivalence_test(prog, "S on q[0]")


def test_CNOT():
    prog = [
        CliffordQRegDecl("q", 2),
        CNOT(CliffordQubitRef("q", 0), CliffordQubitRef("q", 1)),
    ]
    run_equivalence_test(prog, "CNOT(q[0], q[1])")


def test_HCNOTS():
    prog = [
        CliffordQRegDecl("q", 2),
        SingleQubitGate("H", CliffordQubitRef("q", 0)),
        CNOT(CliffordQubitRef("q", 0), CliffordQubitRef("q", 1)),
        SingleQubitGate("S", CliffordQubitRef("q", 1)),
    ]
    run_equivalence_test(prog, "H; CNOT; S")


# ---------------------------------------------------------------------------
# More complicated structured tests
# ---------------------------------------------------------------------------

def test_3qubit_chain():
    """
    3-qubit entangling chain with mixed gates:

        qreg q[3];
        H q[0];
        CNOT q[0], q[1];
        CNOT q[1], q[2];
        S q[0];
        H q[2];
        X q[1];
        Z q[0];
        Y q[2];
    """
    prog = [
        CliffordQRegDecl("q", 3),
        SingleQubitGate("H", CliffordQubitRef("q", 0)),
        CNOT(CliffordQubitRef("q", 0), CliffordQubitRef("q", 1)),
        CNOT(CliffordQubitRef("q", 1), CliffordQubitRef("q", 2)),
        SingleQubitGate("S", CliffordQubitRef("q", 0)),
        SingleQubitGate("H", CliffordQubitRef("q", 2)),
        SingleQubitGate("X", CliffordQubitRef("q", 1)),
        SingleQubitGate("Z", CliffordQubitRef("q", 0)),
        SingleQubitGate("Y", CliffordQubitRef("q", 2)),
    ]
    run_equivalence_test(prog, "3-qubit entangling chain")


def test_teleportation_pattern():
    """
    Teleportation-style Clifford pattern:

        qreg q[3];
        # q[0]: input, q[1], q[2]: ancillas

        H q[1];
        CNOT q[1], q[2];   # create Bell pair between q1 and q2
        CNOT q[0], q[1];
        H q[0];
        S q[2];
        X q[1];
        Z q[0];

    This is a standard teleportation skeleton (without explicit measurement),
    but here we treat it just as a Clifford circuit to stress the compiler.
    """
    prog = [
        CliffordQRegDecl("q", 3),
        SingleQubitGate("H", CliffordQubitRef("q", 1)),
        CNOT(CliffordQubitRef("q", 1), CliffordQubitRef("q", 2)),
        CNOT(CliffordQubitRef("q", 0), CliffordQubitRef("q", 1)),
        SingleQubitGate("H", CliffordQubitRef("q", 0)),
        SingleQubitGate("S", CliffordQubitRef("q", 2)),
        SingleQubitGate("X", CliffordQubitRef("q", 1)),
        SingleQubitGate("Z", CliffordQubitRef("q", 0)),
    ]
    run_equivalence_test(prog, "Teleportation-style pattern")


def test_layered_4qubit_circuit():
    """
    4-qubit layered circuit with several rounds of entangling and 1-qubit Cliffords.
    """
    prog = [CliffordQRegDecl("q", 4)]

    # Layer 1: H on all
    for i in range(4):
        prog.append(SingleQubitGate("H", CliffordQubitRef("q", i)))

    # Layer 2: nearest-neighbor CNOTs
    prog.append(CNOT(CliffordQubitRef("q", 0), CliffordQubitRef("q", 1)))
    prog.append(CNOT(CliffordQubitRef("q", 2), CliffordQubitRef("q", 3)))

    # Layer 3: S and Pauli corrections
    prog.append(SingleQubitGate("S", CliffordQubitRef("q", 1)))
    prog.append(SingleQubitGate("S", CliffordQubitRef("q", 2)))
    prog.append(SingleQubitGate("X", CliffordQubitRef("q", 0)))
    prog.append(SingleQubitGate("Y", CliffordQubitRef("q", 3)))

    # Layer 4: cross CNOTs
    prog.append(CNOT(CliffordQubitRef("q", 1), CliffordQubitRef("q", 2)))
    prog.append(CNOT(CliffordQubitRef("q", 3), CliffordQubitRef("q", 0)))

    # Layer 5: final H on 0 and 3
    prog.append(SingleQubitGate("H", CliffordQubitRef("q", 0)))
    prog.append(SingleQubitGate("H", CliffordQubitRef("q", 3)))

    run_equivalence_test(prog, "Layered 4-qubit Clifford circuit")


def test_pauli_only():
    """
    Long Pauli-only circuit (no measurements) to stress pure frame evolution:

        qreg q[3];
        X q[0]; Z q[1]; Y q[2];
        X q[1]; Y q[0]; Z q[2];
        ...
    """
    prog = [CliffordQRegDecl("q", 3)]
    ops = [("X", 0), ("Z", 1), ("Y", 2), ("X", 1), ("Y", 0), ("Z", 2)]
    # repeat pattern several times
    for _ in range(5):
        for g, q in ops:
            prog.append(SingleQubitGate(g, CliffordQubitRef("q", q)))

    run_equivalence_test(prog, "Pauli-only circuit (no measurements)")


def test_many_HS_repetitions():
    """
    Repeated H and S on same qubit to force many teleportations and ancilla allocations:

        qreg q[1];
        H q[0];
        S q[0];
        H q[0];
        S q[0];
        ...
    """
    prog = [CliffordQRegDecl("q", 1)]
    for _ in range(5):
        prog.append(SingleQubitGate("H", CliffordQubitRef("q", 0)))
        prog.append(SingleQubitGate("S", CliffordQubitRef("q", 0)))

    run_equivalence_test(prog, "Many H/S repetitions on a single qubit")


# ---------------------------------------------------------------------------
# Random tests
# ---------------------------------------------------------------------------

def test_random_circuits_small(num_tests=5, depth=4, num_qubits=2):
    """
    Small random circuits (what we had before), kept for regression.
    """
    gates_1q = ["H", "S", "X", "Y", "Z"]
    gates_2q = ["CNOT"]
    rng = random.Random(999)

    for t in range(num_tests):
        prog = [CliffordQRegDecl("q", num_qubits)]
        for _ in range(depth):
            gtype = rng.choice(["1q", "2q"])
            if gtype == "1q":
                gate = rng.choice(gates_1q)
                q = rng.randrange(num_qubits)
                prog.append(SingleQubitGate(gate, CliffordQubitRef("q", q)))
            else:
                # CNOT with distinct control/target
                c = rng.randrange(num_qubits)
                t2 = (c + 1) % num_qubits
                prog.append(CNOT(CliffordQubitRef("q", c), CliffordQubitRef("q", t2)))

        run_equivalence_test(prog, f"Random small circuit #{t+1}")


def test_random_circuits_large(num_tests=10, depth=10, min_qubits=2, max_qubits=4):
    """
    Larger random test suite:
      - up to 4 qubits
      - depth up to 10
      - mixture of single- and two-qubit gates
    """
    gates_1q = ["H", "S", "X", "Y", "Z"]
    rng = random.Random(2025)

    for t in range(num_tests):
        n = rng.randint(min_qubits, max_qubits)
        prog = [CliffordQRegDecl("q", n)]
        for _ in range(depth):
            gtype = rng.choice(["1q", "2q"])
            if gtype == "1q" or n == 1:
                gate = rng.choice(gates_1q)
                q = rng.randrange(n)
                prog.append(SingleQubitGate(gate, CliffordQubitRef("q", q)))
            else:
                c = rng.randrange(n)
                t2 = rng.randrange(n)
                while t2 == c:
                    t2 = rng.randrange(n)
                prog.append(CNOT(CliffordQubitRef("q", c), CliffordQubitRef("q", t2)))

        run_equivalence_test(prog, f"Random large circuit #{t+1} (n={n}, depth={depth})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Base sanity tests
    # test_H()
    # test_S()
    # test_CNOT()
    # test_HCNOTS()

    # Structured, more complex tests
    test_3qubit_chain()
    test_teleportation_pattern()
    test_layered_4qubit_circuit()
    test_pauli_only()
    test_many_HS_repetitions()

    # Randomized tests
    test_random_circuits_small(num_tests=15, depth=4, num_qubits=10)
    test_random_circuits_large(num_tests=15, depth=10, min_qubits=10, max_qubits=15)

    print("\nAll tests passed.")
