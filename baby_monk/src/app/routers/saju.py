from fastapi import APIRouter, Depends, HTTPException

from app.database.models import MemberSajuInfo
from app.database.repository import MemberSajuInfoRepository
from app.schema.request import CreateMemberSajuInfoRequest
from app.schema.response import MemberSajuInfoSchema

router = APIRouter(prefix="/saju")

# 사주팔자 json으로 받아서 db 저장하기
@router.post("/{member_user_id}/saju_info", status_code=201)
def give_member_saju_word(
        member_user_id: str,
        request: CreateMemberSajuInfoRequest,
        member_saju_repo: MemberSajuInfoRepository = Depends()
) -> MemberSajuInfoSchema:
    if member_saju_repo.is_member_user_id_duplicate(member_user_id):
        raise HTTPException(status_code=400, detail="이미 사주팔자 정보를 저장했습니다.")
    member_saju: MemberSajuInfo = MemberSajuInfo.create_saju(
        user_id=member_user_id,
        request=request
    )
    member_saju: MemberSajuInfo = member_saju_repo.save_saju(member_saju=member_saju)

    return MemberSajuInfoSchema.model_validate(member_saju)




@router.get("/{member_user_id}/saju_text", status_code=200)
def give_text(

):
    pass
