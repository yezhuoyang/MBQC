"""
A simulator for Measurement-Based Quantum Computation (MBQC) in the QMeas language.

- Qubits are tracked via a Pauli frame F : qubit -> {I, X, Y, Z}.
- Measurements are of Pauli observables:
      MX, MY, MZ, MXX, MZZ, MXZ, MYZ, ...
  and store outcomes r ∈ {+1, -1} into classical bits c[i].

- Classical control is via:
      if c[i]==±1 then frame_* ...

- In this first version, each measurement outcome is chosen
  uniformly at random (coin flip), independent of the underlying state.
  The simulator focuses on the structure of measurement patterns
  and Pauli-frame evolution, not exact probabilities.
"""

from clifford import *
from language import *
import qiskit  # kept for future extensions; not used directly here
import random


class Simulator:
    def __init__(self):
        # Program to execute
        self.instructions = []

        # Quantum / classical sizes
        self.num_qubits = 0
        self.num_cbits = 0

        # Pauli frame: qubit_index -> 'I' | 'X' | 'Y' | 'Z'
        self.pauli_frame = {}

        # Classical store: c_index -> outcome in {+1, -1}
        self.classical_store = {}

    # ------------------------------------------------------------
    # Program management
    # ------------------------------------------------------------

    def add_instruction(self, instruction: Instruction):
        self.instructions.append(instruction)

    def load_program(self, instructions):
        self.instructions = list(instructions)

    # ------------------------------------------------------------
    # Pauli-frame helpers
    # ------------------------------------------------------------

    @staticmethod
    def _compose_pauli(p_old: str, p_new: str) -> str:
        """
        Compose Pauli operators in the frame: p_new * p_old.

        We ignore global phases and track only {I, X, Y, Z}.
        The multiplication table modulo phase is:

            I * A = A
            X * X = I, Y * Y = I, Z * Z = I
            X * Y = Z, Y * X = Z
            X * Z = Y, Z * X = Y
            Y * Z = X, Z * Y = X

        (Note: Signs (±i) are discarded.)
        """
        if p_old == "I":
            return p_new
        if p_new == "I":
            return p_old

        if p_old == p_new:
            return "I"

        # Mixed products up to phase
        pair = {p_old, p_new}
        if pair == {"X", "Y"}:
            return "Z"
        if pair == {"X", "Z"}:
            return "Y"
        if pair == {"Y", "Z"}:
            return "X"

        # Fallback (shouldn't happen if we only use I,X,Y,Z)
        return "I"

    def _apply_frame_update(self, op_name: str, qubits):
        """
        Apply a frame_* command to the Pauli frame.

        op_name ∈ {"frame_X", "frame_Y", "frame_Z"}.
        """
        if op_name not in {"frame_X", "frame_Y", "frame_Z"}:
            raise ValueError(f"Unknown frame op: {op_name}")

        pauli = op_name.split("_", 1)[1]  # 'X', 'Y', or 'Z'

        for q in qubits:
            if not isinstance(q, QubitRef):
                raise TypeError(f"Frame op expects QubitRef, got {q}")

            idx = q.index
            old = self.pauli_frame.get(idx, "I")
            new = self._compose_pauli(old, pauli)
            self.pauli_frame[idx] = new

    # ------------------------------------------------------------
    # Measurement helpers
    # ------------------------------------------------------------

    def _simulate_measurement(self, meas: MeasurementInstruction):
        """
        Simulate one Pauli measurement:

            c[k] = M<P> q[...];

        For now:
          - Outcome r ∈ {+1, -1} chosen uniformly at random.
          - We record r in classical_store[c_index].
          - We do not update the underlying stabilizer state explicitly,
            but we could extend this with a full tableau realization.

        This respects the MBQC structure (measure → classical result → frame updates).
        """
        cbit: CBitRef = meas.classical_bit
        c_index = cbit.index

        # Random ±1 outcome
        outcome = random.choice([-1, +1])

        self.classical_store[c_index] = outcome

        print(f"Measure {meas.name} on "
              f"{', '.join(str(q) for q in meas.qubits)} -> {outcome}")

    # ------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------

    def run(self):
        """
        Execute the loaded QMeas program.

        Supported instructions:
          - QRegDecl:  qreg q[N];
          - CRegDecl:  creg c[M];
          - MeasurementInstruction: c[i] = M* ...;
          - FrameInstruction: frame_X(q), frame_Z(q), frame_Y(q);
          - IfInstruction: if c[i]==±1 then FRAME;
        """
        for instr in self.instructions:
            self._execute_instruction(instr)

    def _execute_instruction(self, instr: Instruction):
        # Declarations
        if isinstance(instr, QRegDecl):
            self._handle_qreg_decl(instr)
        elif isinstance(instr, CRegDecl):
            self._handle_creg_decl(instr)

        # Measurement
        elif isinstance(instr, MeasurementInstruction):
            self._simulate_measurement(instr)

        # Frame update
        elif isinstance(instr, FrameInstruction):
            print(f"Frame update: {instr}")
            self._apply_frame_update(instr.name, instr.qubits)

        # Conditional
        elif isinstance(instr, IfInstruction):
            self._execute_if(instr)

        # Fallback: ignore or print
        else:
            print(f"[WARN] Unsupported instruction in simulator: {instr}")

    # ------------------------------------------------------------
    # Instruction handlers
    # ------------------------------------------------------------

    def _handle_qreg_decl(self, decl: QRegDecl):
        self.num_qubits = decl.size
        # Initialize Pauli frame to identity on all qubits
        self.pauli_frame = {i: "I" for i in range(self.num_qubits)}
        print(f"Allocate qreg {decl.reg_name}[{decl.size}]")

    def _handle_creg_decl(self, decl: CRegDecl):
        self.num_cbits = decl.size
        self.classical_store = {}
        print(f"Allocate creg {decl.reg_name}[{decl.size}]")

    def _execute_if(self, if_instr: IfInstruction):
        cbit = if_instr.cbit
        value = if_instr.value
        body = if_instr.body

        actual = self.classical_store.get(cbit.index, None)

        print(f"Check if {cbit}=={value}: current={actual}")

        if actual is None:
            # Condition on an uninitialized classical bit – treat as false.
            return

        if actual == value:
            # Execute the body (currently a single FrameInstruction)
            self._execute_instruction(body)
        else:
            # Skip
            pass


# ---------------------------------------------------------------------
# Example usage (optional smoke test)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # A tiny hand-written QMeas program in your syntax
    prog = [
        QRegDecl("q", 4),
        CRegDecl("c", 3),
        # Example: H gadget on q[0] with ancilla q[1]
        MeasurementInstruction("MZZ", [QubitRef("q", 0), QubitRef("q", 1)], CBitRef("c", 0)),
        MeasurementInstruction("MX", [QubitRef("q", 0)], CBitRef("c", 1)),
        IfInstruction(CBitRef("c", 0), -1, FrameInstruction("frame_Z", [QubitRef("q", 1)])),
        IfInstruction(CBitRef("c", 1), -1, FrameInstruction("frame_X", [QubitRef("q", 1)])),
        # One extra random measurement
        MeasurementInstruction("MXX", [QubitRef("q", 2), QubitRef("q", 3)], CBitRef("c", 2)),
    ]

    sim = Simulator()
    sim.load_program(prog)
    sim.run()

    print("\nFinal classical store:", sim.classical_store)
    print("Final Pauli frame:", sim.pauli_frame)
