"""
Compiler from Clifford IR to measurement-based QMeas programs.

Implements the gadgets from the paper:

- H gate:
    r1 := MZZ(q, a);
    r2 := MX(q);
    if r1 = -1 then frame_Z(a);
    if r2 = -1 then frame_X(a);
    discard q.

- S gate:
    r1 := MYZ(q, a);
    r2 := MX(q);
    if r1 = -1 then frame_Z(a);
    if r2 = -1 then frame_X(a);
    discard q.

- CNOT(c, t):
    r1 := MZZ(c, a1);
    r2 := MXX(a1, t);
    r3 := MZZ(a1, a2);
    r4 := MX(a1);
    if r1 = -1 then frame_Z(t);
    if r2 = -1 then frame_X(c);
    if r3 = -1 then frame_Z(c);
    if r4 = -1 then frame_X(t);
    discard a1, a2.

Pauli X/Y/Z are compiled to pure Pauli-frame updates:
    X(q) -> frame_X(q);
    Y(q) -> frame_Y(q);
    Z(q) -> frame_Z(q);
(no measurements needed)
"""

from typing import List, Dict, Tuple

import language
import clifford


LogicalKey = Tuple[str, int]  # (reg_name, logical_index)


class MBQCCompiler:
    """
    Compile a Clifford program to a QMeas (measurement-based) program.

    Assumptions:
      - There is a single quantum register declaration in the Clifford program:
            qreg q[N];
      - Clifford gates are drawn from {H, S, CNOT, X, Y, Z}.
      - The target language is QMeas (language.py): only
            qreg, creg, measurements M*, and frame_* / if ... then frame_*.
    """

    def __init__(self, clifford_program: List[clifford.CliffordInstruction]):
        self.program = clifford_program

        # Will be filled in compile()
        self.compiled_program: List[language.Instruction] = []

        # Logical -> physical qubit mapping:
        #   key:  (reg_name, logical_index)
        #   value: language.QubitRef(reg_name, physical_index)
        self.logical_to_physical: Dict[LogicalKey, language.QubitRef] = {}

        # Next available ancilla index in the quantum register
        self.next_ancilla_index: int = 0

        # Next available classical bit index in creg c[]
        self.next_c_index: int = 0

        # Base register info
        self.base_reg_name: str = "q"
        self.base_reg_size: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self):
        """Top-level compilation driver."""
        self.compiled_program = []

        qreg_decls: List[clifford.QRegDecl] = []
        gates: List[clifford.CliffordInstruction] = []

        # Separate qreg declarations from actual gates
        for instr in self.program:
            if isinstance(instr, clifford.QRegDecl):
                qreg_decls.append(instr)
            else:
                gates.append(instr)

        # For now, support exactly one quantum register
        if len(qreg_decls) != 1:
            raise ValueError(
                "MBQCCompiler currently expects exactly one 'qreg' declaration"
            )

        base_qreg = qreg_decls[0]
        self.base_reg_name = base_qreg.reg_name
        self.base_reg_size = base_qreg.size

        # Initialize logical -> physical mapping:
        # initially logical i is stored at physical i.
        self.logical_to_physical = {
            (self.base_reg_name, i): language.QubitRef(self.base_reg_name, i)
            for i in range(self.base_reg_size)
        }

        # Pre-compute resources: how many ancillas and measurements?
        ancilla_count, meas_count = self._precompute_resources(gates)

        # Configure ancilla and classical indices
        self.next_ancilla_index = self.base_reg_size
        self.next_c_index = 0

        total_qubits = self.base_reg_size + ancilla_count
        total_cbits = meas_count

        # Emit updated qreg and creg declarations
        self.compiled_program.append(
            language.QRegDecl(self.base_reg_name, total_qubits)
        )
        if total_cbits > 0:
            self.compiled_program.append(
                language.CRegDecl("c", total_cbits)
            )

        # Compile each Clifford gate into QMeas instructions
        for gate in gates:
            self.compiled_program.extend(self._compile_gate(gate))

    def get_compiled_program(self) -> List[language.Instruction]:
        return self.compiled_program

    # ------------------------------------------------------------------
    # Resource analysis
    # ------------------------------------------------------------------

    def _precompute_resources(
        self,
        gates: List[clifford.CliffordInstruction],
    ) -> Tuple[int, int]:
        """
        Determine how many ancilla qubits and measurement outcomes we need
        for the entire Clifford program.

        From the paper gadgets:
          - H: 1 ancilla, 2 measurements (r1, r2)
          - S: 1 ancilla, 2 measurements
          - CNOT: 2 ancillas, 4 measurements
          - X/Y/Z: 0 ancillas, 0 measurements (pure frame updates)
        """
        ancillas = 0
        measurements = 0

        for instr in gates:
            if not isinstance(instr, clifford.CliffordInstruction):
                continue

            name = instr.name

            if name == "H":
                ancillas += 1
                measurements += 2
            elif name == "S":
                ancillas += 1
                measurements += 2
            elif name == "CNOT":
                ancillas += 2
                measurements += 4
            elif name in {"X", "Y", "Z"}:
                # Pauli gates compiled to frame updates only
                continue
            else:
                raise ValueError(f"Unsupported Clifford gate: {name!r}")

        return ancillas, measurements

    # ------------------------------------------------------------------
    # Helpers for fresh references
    # ------------------------------------------------------------------

    def _fresh_ancilla(self) -> language.QubitRef:
        """Allocate a fresh ancilla qubit on the same base register."""
        q = language.QubitRef(self.base_reg_name, self.next_ancilla_index)
        self.next_ancilla_index += 1
        return q

    def _fresh_cbit(self) -> language.CBitRef:
        """Allocate a fresh classical bit c[i] for a measurement outcome."""
        c = language.CBitRef("c", self.next_c_index)
        self.next_c_index += 1
        return c

    def _lookup_phys(self, q: clifford.CliffordQubitRef) -> language.QubitRef:
        """
        Given a Clifford qubit reference q (reg_name, index),
        return the current physical location in QMeas space
        from the logical-to-physical map.
        """
        key: LogicalKey = (q.reg_name, q.index)
        if key not in self.logical_to_physical:
            raise KeyError(
                f"Logical qubit {q.reg_name}[{q.index}] not in mapping"
            )
        return self.logical_to_physical[key]

    # ------------------------------------------------------------------
    # Gate compilation
    # ------------------------------------------------------------------

    def _compile_gate(
        self,
        instr: clifford.CliffordInstruction,
    ) -> List[language.Instruction]:
        """
        Compile a single Clifford gate to a QMeas snippet,
        according to the gadgets in the paper.
        """
        if not isinstance(instr, clifford.CliffordInstruction):
            return []

        name = instr.name

        if name == "H":
            return self._compile_H(instr)
        elif name == "S":
            return self._compile_S(instr)
        elif name == "CNOT":
            return self._compile_CNOT(instr)
        elif name in {"X", "Y", "Z"}:
            return self._compile_Pauli(instr)
        else:
            raise ValueError(f"Unsupported Clifford gate: {name!r}")

    def _compile_H(
        self,
        instr: clifford.CliffordInstruction,
    ) -> List[language.Instruction]:
        """
        H gate gadget:

            r1 := MZZ(q, a);
            r2 := MX(q);
            if r1 = -1 then frame_Z(a);
            if r2 = -1 then frame_X(a);
            discard q.

        Logical qubit is teleported from old location of q to ancilla a.
        """
        if len(instr.qubits) != 1:
            raise ValueError("H gate must act on exactly 1 qubit")

        q_log = instr.qubits[0]
        q_phys = self._lookup_phys(q_log)

        # Allocate ancilla 'a'
        a = self._fresh_ancilla()

        # Classical outcomes r1, r2 become c[i], c[i+1]
        r1 = self._fresh_cbit()
        r2 = self._fresh_cbit()

        res: List[language.Instruction] = []

        # r1 := MZZ(q, a);
        res.append(language.MeasurementInstruction("MZZ", [q_phys, a], r1))

        # r2 := MX(q);
        res.append(language.MeasurementInstruction("MX", [q_phys], r2))

        # if r1 = -1 then frame_Z(a);
        res.append(
            language.IfInstruction(
                cbit=r1,
                value=-1,
                body=language.FrameInstruction("frame_Z", [a]),
            )
        )

        # if r2 = -1 then frame_X(a);
        res.append(
            language.IfInstruction(
                cbit=r2,
                value=-1,
                body=language.FrameInstruction("frame_X", [a]),
            )
        )

        # Update logical location: logical q now lives on ancilla 'a'
        key: LogicalKey = (q_log.reg_name, q_log.index)
        self.logical_to_physical[key] = a

        # 'discard q' is logical; we simply don't use q_phys again.
        return res

    def _compile_S(
        self,
        instr: clifford.CliffordInstruction,
    ) -> List[language.Instruction]:
        """
        S gate gadget:

            r1 := MYZ(q, a);
            r2 := MX(q);
            if r1 = -1 then frame_Z(a);
            if r2 = -1 then frame_X(a);
            discard q.

        Logical qubit is teleported from old location of q to ancilla a.
        """
        if len(instr.qubits) != 1:
            raise ValueError("S gate must act on exactly 1 qubit")

        q_log = instr.qubits[0]
        q_phys = self._lookup_phys(q_log)

        a = self._fresh_ancilla()
        r1 = self._fresh_cbit()
        r2 = self._fresh_cbit()

        res: List[language.Instruction] = []

        # r1 := MYZ(q, a);
        res.append(language.MeasurementInstruction("MYZ", [q_phys, a], r1))

        # r2 := MX(q);
        res.append(language.MeasurementInstruction("MX", [q_phys], r2))

        # if r1 = -1 then frame_Z(a);
        res.append(
            language.IfInstruction(
                cbit=r1,
                value=-1,
                body=language.FrameInstruction("frame_Z", [a]),
            )
        )

        # if r2 = -1 then frame_X(a);
        res.append(
            language.IfInstruction(
                cbit=r2,
                value=-1,
                body=language.FrameInstruction("frame_X", [a]),
            )
        )

        # Logical q now lives on 'a'
        key: LogicalKey = (q_log.reg_name, q_log.index)
        self.logical_to_physical[key] = a

        return res

    def _compile_CNOT(
        self,
        instr: clifford.CliffordInstruction,
    ) -> List[language.Instruction]:
        """
        CNOT gadget (control c, target t):

            r1 := MZZ(c, a1);
            r2 := MXX(a1, t);
            r3 := MZZ(a1, a2);
            r4 := MX(a1);
            if r1 = -1 then frame_Z(t);
            if r2 = -1 then frame_X(c);
            if r3 = -1 then frame_Z(c);
            if r4 = -1 then frame_X(t);
            discard a1, a2.

        Logical qubits remain on the same physical wires (c_phys, t_phys).
        """
        if len(instr.qubits) != 2:
            raise ValueError("CNOT must act on exactly 2 qubits")

        c_log = instr.qubits[0]
        t_log = instr.qubits[1]

        c_phys = self._lookup_phys(c_log)
        t_phys = self._lookup_phys(t_log)

        # Ancillas a1, a2
        a1 = self._fresh_ancilla()
        a2 = self._fresh_ancilla()

        # Measurement outcomes r1..r4
        r1 = self._fresh_cbit()
        r2 = self._fresh_cbit()
        r3 = self._fresh_cbit()
        r4 = self._fresh_cbit()

        res: List[language.Instruction] = []

        # r1 := MZZ(c, a1);
        res.append(language.MeasurementInstruction("MZZ", [c_phys, a1], r1))

        # r2 := MXX(a1, t);
        res.append(language.MeasurementInstruction("MXX", [a1, t_phys], r2))

        # r3 := MZZ(a1, a2);
        res.append(language.MeasurementInstruction("MZZ", [a1, a2], r3))

        # r4 := MX(a1);
        res.append(language.MeasurementInstruction("MX", [a1], r4))

        # if r1 = -1 then frame_Z(t);
        res.append(
            language.IfInstruction(
                cbit=r1,
                value=-1,
                body=language.FrameInstruction("frame_Z", [t_phys]),
            )
        )

        # if r2 = -1 then frame_X(c);
        res.append(
            language.IfInstruction(
                cbit=r2,
                value=-1,
                body=language.FrameInstruction("frame_X", [c_phys]),
            )
        )

        # if r3 = -1 then frame_Z(c);
        res.append(
            language.IfInstruction(
                cbit=r3,
                value=-1,
                body=language.FrameInstruction("frame_Z", [c_phys]),
            )
        )

        # if r4 = -1 then frame_X(t);
        res.append(
            language.IfInstruction(
                cbit=r4,
                value=-1,
                body=language.FrameInstruction("frame_X", [t_phys]),
            )
        )

        # Logical locations for c,t do NOT change for this gadget.
        return res

    def _compile_Pauli(
        self,
        instr: clifford.CliffordInstruction,
    ) -> List[language.Instruction]:
        """
        Compile Pauli X/Y/Z gates as pure Pauli-frame updates:

            X(q) -> frame_X(q_phys);
            Y(q) -> frame_Y(q_phys);
            Z(q) -> frame_Z(q_phys);

        No measurements or ancillas are used.
        """
        if len(instr.qubits) != 1:
            raise ValueError("Pauli gate must act on exactly 1 qubit")

        q_log = instr.qubits[0]
        q_phys = self._lookup_phys(q_log)

        gate = instr.name
        if gate == "X":
            name = "frame_X"
        elif gate == "Y":
            name = "frame_Y"
        elif gate == "Z":
            name = "frame_Z"
        else:
            raise ValueError(f"Unexpected Pauli gate {gate!r} in _compile_Pauli")

        return [language.FrameInstruction(name, [q_phys])]


if __name__ == "__main__":
    # Small smoke test
    example_clifford = [
        clifford.QRegDecl("q", 3),
        clifford.CNOT(
            clifford.CliffordQubitRef("q", 0),
            clifford.CliffordQubitRef("q", 1),
        ),
        clifford.SingleQubitGate("H", clifford.CliffordQubitRef("q", 2)),
        clifford.SingleQubitGate("S", clifford.CliffordQubitRef("q", 1)),
        clifford.SingleQubitGate("X", clifford.CliffordQubitRef("q", 0)),
    ]

    compiler = MBQCCompiler(example_clifford)
    compiler.compile()

    for instr in compiler.get_compiled_program():
        print(instr)
