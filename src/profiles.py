"""Predefined student profiles for testing personalized advising."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StudentProfile:
    name: str
    student_id: str
    current_semester: int
    passed_courses: dict[str, float] = field(default_factory=dict)
    failed_courses: dict[str, float] = field(default_factory=dict)
    description: str = ""

    def summary(self) -> str:
        """Build a natural-language context block for the agent prompt."""
        lines = [
            f"Student: {self.name} (ID: {self.student_id})",
            f"Current semester: {self.current_semester}",
            f"Passed courses ({len(self.passed_courses)}):",
        ]
        for code, grade in sorted(self.passed_courses.items()):
            lines.append(f"  - {code}: {grade:.1f}/10")

        if self.failed_courses:
            lines.append(f"Failed courses ({len(self.failed_courses)}):")
            for code, grade in sorted(self.failed_courses.items()):
                lines.append(f"  - {code}: {grade:.1f}/10 (FAILED, must retake)")

        gpa = self.gpa
        if gpa is not None:
            lines.append(f"Current GPA: {gpa:.2f}/10")

        return "\n".join(lines)

    @property
    def gpa(self) -> float | None:
        if not self.passed_courses:
            return None
        return sum(self.passed_courses.values()) / len(self.passed_courses)


PROFILES: dict[str, StudentProfile] = {
    "none": StudentProfile(
        name="Anonymous",
        student_id="N/A",
        current_semester=0,
        description="No profile — general questions only",
    ),
    "freshman_s1": StudentProfile(
        name="Nguyen Van An",
        student_id="SE180001",
        current_semester=1,
        description="Freshman, just started semester 1",
        passed_courses={
            "OTP101": 8.0,
            "PEN": 7.5,
        },
    ),
    "sophomore_s3": StudentProfile(
        name="Tran Thi Bao",
        student_id="SE180042",
        current_semester=3,
        description="Sophomore at semester 3, solid GPA",
        passed_courses={
            "OTP101": 8.0,
            "PEN": 7.5,
            "CSI106": 8.5,
            "MAD101": 7.0,
            "MAE101": 8.0,
            "PFP191": 9.0,
            "SSA101": 7.5,
            "AIG202c": 8.0,
            "CEA201": 7.0,
            "CSD203": 8.5,
            "DBI202": 7.5,
            "JPD113": 6.5,
        },
    ),
    "junior_s5": StudentProfile(
        name="Le Hoang Minh",
        student_id="SE180103",
        current_semester=5,
        description="Junior at semester 5, wants to specialize in AI",
        passed_courses={
            "OTP101": 7.0,
            "PEN": 6.5,
            "CSI106": 7.5,
            "MAD101": 6.0,
            "MAE101": 7.0,
            "PFP191": 8.0,
            "SSA101": 7.0,
            "AIG202c": 7.5,
            "CEA201": 6.5,
            "CSD203": 7.0,
            "DBI202": 6.5,
            "JPD113": 5.5,
            "ADY201m": 7.5,
            "ITE303c": 8.0,
            "JPD123": 6.0,
            "MAI391": 7.0,
            "MAS291": 6.5,
            "AIL303m": 7.5,
            "CPV301": 7.0,
            "DAP391m": 8.0,
            "SSG105": 7.5,
            "SWE201c": 7.0,
        },
    ),
    "struggling_s4": StudentProfile(
        name="Pham Duc Huy",
        student_id="SE180200",
        current_semester=4,
        description="Semester 4, failed MAS291, low GPA in math courses",
        passed_courses={
            "OTP101": 6.0,
            "PEN": 5.5,
            "CSI106": 6.5,
            "MAD101": 4.5,
            "MAE101": 5.0,
            "PFP191": 7.0,
            "SSA101": 6.0,
            "AIG202c": 6.0,
            "CEA201": 5.5,
            "CSD203": 6.5,
            "DBI202": 6.0,
            "JPD113": 5.0,
            "ADY201m": 6.0,
            "ITE303c": 7.0,
            "JPD123": 5.0,
            "MAI391": 5.0,
        },
        failed_courses={
            "MAS291": 3.5,
        },
    ),
}
