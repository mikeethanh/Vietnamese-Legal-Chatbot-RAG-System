"""
Legal tools for Vietnamese law chatbot agent
Simple, practical tools that don't require complex dependencies
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def calculate_contract_penalty(
    contract_value: float, penalty_rate: float, days_late: int
) -> Dict:
    """
    Tính tiền phạt vi phạm hợp đồng theo quy định pháp luật Việt Nam.

    Args:
        contract_value: Giá trị hợp đồng (VNĐ)
        penalty_rate: Tỷ lệ phạt (%, thường 0.05-0.3%/ngày theo Bộ luật Dân sự)
        days_late: Số ngày chậm trễ

    Returns:
        Dict with penalty amount and details
    """
    try:
        penalty_amount = contract_value * (penalty_rate / 100) * days_late

        # Kiểm tra giới hạn phạt (tối đa 8-12% giá trị hợp đồng theo thông lệ)
        max_penalty = contract_value * 0.12
        if penalty_amount > max_penalty:
            penalty_amount = max_penalty
            note = "Đã áp dụng mức phạt tối đa 12% giá trị hợp đồng"
        else:
            note = "Tính theo tỷ lệ phạt đã thỏa thuận"

        result = {
            "contract_value": f"{contract_value:,.0f} VNĐ",
            "penalty_rate": f"{penalty_rate}%/ngày",
            "days_late": days_late,
            "penalty_amount": f"{penalty_amount:,.0f} VNĐ",
            "note": note,
        }

        logger.info(f"[TOOL] Contract penalty calculated: {result}")
        return result

    except Exception as e:
        logger.error(f"Error calculating contract penalty: {e}")
        return {"error": str(e)}


def check_legal_entity_age(birth_year: int, action_type: str = "sign_contract") -> Dict:
    """
    Kiểm tra tuổi pháp lý để thực hiện hành vi dân sự theo Bộ luật Dân sự Việt Nam.

    Args:
        birth_year: Năm sinh
        action_type: Loại hành vi (sign_contract, marriage, work, criminal_responsibility)

    Returns:
        Dict with eligibility status and legal details
    """
    try:
        current_year = datetime.now().year
        age = current_year - birth_year

        # Các mốc tuổi pháp lý theo pháp luật Việt Nam
        age_requirements = {
            "sign_contract": {
                "min_age": 18,
                "description": "Đủ 18 tuổi để ký hợp đồng (Điều 21 Bộ luật Dân sự 2015)",
                "partial_age": 15,
                "partial_note": "Từ 15-18 tuổi cần có sự đồng ý của người đại diện hợp pháp",
            },
            "marriage": {
                "min_age": 18,  # Nam 20, Nữ 18
                "description": "Nam đủ 20 tuổi, Nữ đủ 18 tuổi (Điều 8 Luật Hôn nhân và Gia đình 2014)",
                "note": "Nam: 20 tuổi, Nữ: 18 tuổi",
            },
            "work": {
                "min_age": 15,
                "description": "Đủ 15 tuổi được làm việc (Điều 143 Bộ luật Lao động 2019)",
                "note": "Dưới 15 tuổi chỉ làm công việc nghệ thuật với điều kiện đặc biệt",
            },
            "criminal_responsibility": {
                "min_age": 16,
                "description": "Đủ 16 tuổi chịu trách nhiệm hình sự (Điều 12 Bộ luật Hình sự 2015)",
                "partial_age": 14,
                "partial_note": "Từ 14-16 tuổi chỉ chịu trách nhiệm với tội đặc biệt nghiêm trọng",
            },
        }

        req = age_requirements.get(action_type, age_requirements["sign_contract"])
        min_age = req["min_age"]
        partial_age = req.get("partial_age", 0)

        if age >= min_age:
            eligible = True
            status = "Đủ điều kiện"
        elif partial_age > 0 and age >= partial_age:
            eligible = "partial"
            status = f"Có điều kiện: {req.get('partial_note', '')}"
        else:
            eligible = False
            status = "Chưa đủ điều kiện"

        result = {
            "age": age,
            "action_type": action_type,
            "eligible": eligible,
            "status": status,
            "legal_basis": req["description"],
            "note": req.get("note", ""),
        }

        logger.info(f"[TOOL] Legal age check: {result}")
        return result

    except Exception as e:
        logger.error(f"Error checking legal age: {e}")
        return {"error": str(e)}


def calculate_inheritance_share(total_value: float, heirs: List[Dict]) -> Dict:
    """
    Tính phần thừa kế theo pháp luật Việt Nam (thừa kế theo pháp luật, hàng thừa kế thứ nhất).

    Args:
        total_value: Tổng giá trị tài sản thừa kế (VNĐ)
        heirs: Danh sách người thừa kế [{"name": str, "relation": str, "is_minor": bool}]
               relation: spouse, child, parent

    Returns:
        Dict with inheritance shares for each heir
    """
    try:
        # Theo Điều 651 Bộ luật Dân sự 2015
        # Hàng thừa kế thứ nhất chia đều: vợ/chồng, con, cha mẹ

        if not heirs:
            return {"error": "Không có người thừa kế"}

        # Đếm số người thừa kế
        num_heirs = len(heirs)
        share_per_heir = total_value / num_heirs

        result = {
            "total_value": f"{total_value:,.0f} VNĐ",
            "num_heirs": num_heirs,
            "share_per_heir": f"{share_per_heir:,.0f} VNĐ",
            "distribution": [],
            "legal_basis": "Điều 651 Bộ luật Dân sự 2015 - Người thừa kế hàng thứ nhất chia đều",
            "note": "Áp dụng cho thừa kế theo pháp luật, hàng thừa kế thứ nhất",
        }

        for heir in heirs:
            heir_share = {
                "name": heir.get("name", ""),
                "relation": heir.get("relation", ""),
                "share": f"{share_per_heir:,.0f} VNĐ",
                "percentage": f"{(100/num_heirs):.2f}%",
            }

            # Lưu ý đặc biệt cho người chưa thành niên
            if heir.get("is_minor", False):
                heir_share["note"] = (
                    "Người chưa thành niên, cần người đại diện quản lý tài sản"
                )

            result["distribution"].append(heir_share)

        logger.info(f"[TOOL] Inheritance calculated: {result}")
        return result

    except Exception as e:
        logger.error(f"Error calculating inheritance: {e}")
        return {"error": str(e)}


def check_business_name_rules(business_name: str) -> Dict:
    """
    Kiểm tra tên doanh nghiệp có hợp lệ theo Luật Doanh nghiệp Việt Nam không.

    Args:
        business_name: Tên doanh nghiệp cần kiểm tra

    Returns:
        Dict with validation results
    """
    try:
        issues = []
        warnings = []

        # Theo Điều 36 Luật Doanh nghiệp 2020

        # 1. Không được trùng hoặc tương tự với tên đã đăng ký
        # (Cần tra cứu cơ sở dữ liệu - bỏ qua trong demo đơn giản)

        # 2. Không chứa các từ ngữ cấm
        prohibited_words = [
            "việt nam",
            "quốc gia",
            "chính phủ",
            "nhà nước",
            "đảng",
            "bộ",
            "ban",
            "ngành",
            "ủy ban",
        ]

        name_lower = business_name.lower()
        for word in prohibited_words:
            if word in name_lower:
                issues.append(
                    f"Tên không được chứa từ '{word}' (Điều 36 Luật Doanh nghiệp 2020)"
                )

        # 3. Kiểm tra độ dài (không quá ngắn)
        if len(business_name) < 5:
            warnings.append("Tên doanh nghiệp quá ngắn, nên có ít nhất 5 ký tự")

        # 4. Kiểm tra ký tự đặc biệt
        import re

        if re.search(r'[!@#$%^&*()_+=\[\]{};\'\\:"|,.<>?/~`]', business_name):
            warnings.append(
                "Tên chứa ký tự đặc biệt, có thể gây khó khăn trong đăng ký"
            )

        # 5. Kiểm tra số ở đầu
        if business_name and business_name[0].isdigit():
            warnings.append("Tên bắt đầu bằng số, nên bắt đầu bằng chữ cái")

        is_valid = len(issues) == 0

        result = {
            "business_name": business_name,
            "is_valid": is_valid,
            "status": "Hợp lệ" if is_valid else "Không hợp lệ",
            "issues": issues,
            "warnings": warnings,
            "legal_basis": "Điều 36 Luật Doanh nghiệp 2020",
            "recommendation": "Nên tra cứu trên hệ thống đăng ký kinh doanh quốc gia để đảm bảo không trùng lặp",
        }

        logger.info(f"[TOOL] Business name check: {result}")
        return result

    except Exception as e:
        logger.error(f"Error checking business name: {e}")
        return {"error": str(e)}


def get_statute_of_limitations(case_type: str) -> Dict:
    """
    Tra cứu thời hiệu khởi kiện theo pháp luật Việt Nam.

    Args:
        case_type: Loại vụ việc (civil, labor, administrative, criminal)

    Returns:
        Dict with statute of limitations info
    """
    try:
        # Theo các bộ luật Việt Nam
        statutes = {
            "civil": {
                "general": "3 năm",
                "description": "Thời hiệu khởi kiện chung là 3 năm (Điều 155 Bộ luật Dân sự 2015)",
                "exceptions": [
                    "Tranh chấp quyền sở hữu đất đai: 10 năm",
                    "Yêu cầu công nhận cha, mẹ, con: Không có thời hiệu",
                    "Bồi thường thiệt hại ngoài hợp đồng: 3 năm",
                ],
                "start_date": "Từ ngày người có quyền yêu cầu biết hoặc phải biết quyền và lợi ích hợp pháp bị xâm phạm",
            },
            "labor": {
                "general": "1 năm",
                "description": "Thời hiệu khởi kiện tranh chấp lao động là 1 năm (Điều 193 Bộ luật Lao động 2019)",
                "exceptions": [
                    "Tranh chấp về tiền lương, trợ cấp thôi việc: 2 năm",
                    "Sa thải trái pháp luật: 1 năm",
                ],
                "start_date": "Từ ngày quyền, lợi ích bị xâm phạm",
            },
            "administrative": {
                "general": "1 năm",
                "description": "Thời hiệu khởi kiện hành chính là 1 năm (Điều 31 Luật TTHC 2015)",
                "exceptions": [
                    "Quyết định xử phạt vi phạm hành chính: 1 năm",
                    "Quyết định thu hồi đất: 1 năm",
                ],
                "start_date": "Từ ngày nhận được quyết định hành chính hoặc biết quyết định",
            },
            "criminal": {
                "general": "Tùy mức hình phạt",
                "description": "Thời hiệu truy cứu trách nhiệm hình sự (Điều 27 Bộ luật Hình sự 2015)",
                "details": [
                    "Phạm tội ít nghiêm trọng: 5 năm",
                    "Phạm tội nghiêm trọng: 10 năm",
                    "Phạm tội rất nghiêm trọng: 15 năm",
                    "Phạm tội đặc biệt nghiêm trọng: 20 năm",
                    "Tội phạm chiến tranh, tội phạm chống loài người: Không có thời hiệu",
                ],
                "start_date": "Từ ngày thực hiện tội phạm",
            },
        }

        statute = statutes.get(case_type)
        if not statute:
            return {
                "error": f"Loại vụ việc '{case_type}' không hợp lệ",
                "valid_types": list(statutes.keys()),
            }

        result = {
            "case_type": case_type,
            "general_statute": statute["general"],
            "description": statute["description"],
            "start_date": statute["start_date"],
        }

        if "exceptions" in statute:
            result["exceptions"] = statute["exceptions"]
        if "details" in statute:
            result["details"] = statute["details"]

        logger.info(f"[TOOL] Statute of limitations: {result}")
        return result

    except Exception as e:
        logger.error(f"Error getting statute of limitations: {e}")
        return {"error": str(e)}
