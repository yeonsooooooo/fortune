from datetime import datetime, timedelta

import bcrypt
from jose import jwt


class MemberService:
    encoding: str = "UTF-8"
    secret_key: str = "448a417d4e67c912760a6a146d1bdfceef0432761bc01df962cfff2983516cf7"
    jwt_algorithm: str = "HS256"

    def hash_password(self, plain_password: str) -> str:
        hashed_password: bytes = bcrypt.hashpw(
            plain_password.encode(self.encoding),
            salt=bcrypt.gensalt(),
        )
        return hashed_password.decode(self.encoding)

    def verify_password(
            self, plain_password: str, hashed_password: str
    ) -> bool:
        return bcrypt.checkpw(
            plain_password.encode(self.encoding),
            hashed_password.encode(self.encoding)
        )

    def create_jwt(self, member_user_id: str) -> str:
        return jwt.encode(
            {
                "sub": member_user_id,
                "exp": datetime.now() + timedelta(days=1)
            },
            self.secret_key,
            algorithm=self.jwt_algorithm,
        )

    def decode_jwt(self, access_token: str) -> str:
        payload: dict = jwt.decode(
            access_token, self.secret_key, algorithms=[self.jwt_algorithm]
        )

        return payload["sub"]
