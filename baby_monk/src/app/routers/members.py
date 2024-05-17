from fastapi import HTTPException, Depends, APIRouter

from fastapi import HTTPException, Depends, APIRouter

from app.database.models import Member
from app.database.repository import MemberRepository
from app.schema.request import CreateMemberRequest, UpdateMemberRequest, LogInRequest
from app.schema.response import MemberSchema, JWTResponse
from app.service.member import MemberService

router = APIRouter(prefix="/members")


@router.post("", status_code=201)
def create_members_handler(
        request: CreateMemberRequest,
        member_repo: MemberRepository = Depends(),
        member_service: MemberService = Depends()
) -> MemberSchema:
    if member_repo.is_member_user_id_duplicate(request.user_id):
        raise HTTPException(status_code=400, detail="이미 사용 중인 아이디입니다.")

    hashed_password: str = member_service.hash_password(
        plain_password=request.password
    )

    member: Member = Member.create(
        request=request, password=hashed_password)  # id = None
    member: Member = member_repo.create_member(member=member)  # id 확정

    return MemberSchema.model_validate(member)


# 전체 멤버 조회 -> 필요하면 활성화
# @router.get("", status_code=200)
# def get_members_handler(
#         member_repo: MemberRepository = Depends(),
#         access_token: str = Depends(get_access_token)
# ):
#     members: List[Member] = member_repo.get_members()
#     return MemberListSchema(
#         members=[MemberSchema.model_validate(member) for member in members]
#     )

# 사용자의 아이디를 받아서 개인 정보를 받아오는 것. 토큰 활용 예정
@router.get("/{member_user_id}", status_code=200)
def get_member_handler(
        member_user_id: str,
        member_repo: MemberRepository = Depends(),
) -> MemberSchema:
    member: Member | None = member_repo.get_member_by_member_user_id(member_user_id=member_user_id)
    if member:
        return MemberSchema.model_validate(member)
    raise HTTPException(status_code=404, detail="해당 회원을 조회할 수 없습니다.")

# 사용자의 아이디를 통해서 개인정보를 수정하는 것. 토큰 활용 예정
@router.patch("/{member_user_id}", status_code=200)
def update_member_handler(
        member_user_id: str,
        request: UpdateMemberRequest,
        member_repo: MemberRepository = Depends(),
):
    update_data = request.dict(exclude_unset=True)
    member: Member | None = member_repo.get_member_by_member_user_id(member_user_id=member_user_id)
    if member:
        updated_member = member_repo.update_member(member_user_id=member_user_id, member=member,
                                                   update_data=update_data)
        return MemberSchema.model_validate(updated_member)
    raise HTTPException(status_code=404, detail="해당 회원을 조회할 수 없습니다.")

# 회원가입 시 아이디가 중복되는지 확인하는 것. 중복 버튼 누른 후 해당 api 활용될 예정
@router.get("/check-duplicate/{member_user_id}", status_code=200)
def check_member_user_id_duplicate(
        member_user_id: str,
        member_repo: MemberRepository = Depends()
) -> dict:
    is_duplicate = member_repo.is_member_user_id_duplicate(member_user_id=member_user_id)

    if is_duplicate:
        raise HTTPException(status_code=409, detail="해당 아이디는 이미 사용중입니다.")

    return {"message": "사용 가능한 아이디입니다."}

# 로그인 시 사용될 api -> 인증 토큰 + refresh 토큰 생성해서 진행할 예정
@router.post("/log-in")
def member_log_in_handler(
        request: LogInRequest,
        member_repo: MemberRepository = Depends(),
        member_service: MemberService = Depends()
):
    member: Member | None = member_repo.get_member_by_member_user_id(member_user_id=request.member_user_id)

    if not member:
        raise HTTPException(status_code=404, detail="회원 정보가 없습니다.")

    verified: bool = member_service.verify_password(
        plain_password=request.password,
        hashed_password=member.password
    )

    if not verified:
        raise HTTPException(status_code=401, detail="비밀번호가 틀렸습니다.")

    access_token: str = member_service.create_jwt(member_user_id=member.user_id)

    return JWTResponse(access_token=access_token)
