from app.database.models import Member


def test_get_members(mocker, client):
    # mocker.patch.object(MemberRepository,"get_members", return_value=[
    #     Member(id=1, user_id="hy010827", email="hy010827@naver.com", password="1234", name="함윤", gender="male",
    #            birth_date="2001-8-27", birth_time="09:00:00")
    # ])
    response = client.get("/members")
    assert response.status_code == 200
    assert response.json() == {
        "members": [
            {"id": 1, "user_id": "hy010827", "email": "hy010827@naver.com", "password": "1234",
             "name": "함윤", "gender": "M", "birth_date": "2001-08-27", "birth_time": "09:00:00"}
        ]
    }


def test_get_member(client, mocker):
    # # 200
    # mocker.patch(MemberRepository,"get_member_by_member_id",
    #              return_value=Member(id=1, user_id="hy010827", email="hy010827@naver.com", password="1234", name="함윤",
    #                                  gender="male", birth_date="2001-8-27", birth_time="09:00:00"))
    response = client.get("/members/1")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "user_id": "hy010827", "email": "hy010827@naver.com", "password": "1234",
                               "name": "함윤", "gender": "M", "birth_date": "2001-08-27", "birth_time": "09:00:00"}

    # 404
    # mocker.patch(MemberRepository,"get_todos", return_value=None)
    response = client.get("/members/1")
    assert response.status_code == 200
    assert response.json() == {"id": 1, "user_id": "hy010827", "email": "hy010827@naver.com", "password": "1234",
                               "name": "함윤", "gender": "M", "birth_date": "2001-08-27", "birth_time": "09:00:00"}


def test_create_member(client, mocker):
    mocker.patch("app.database.repository.MemberRepository.create_member",
                 return_value=Member(id=2, user_id="test", email="test", password="1234", name="test_record",
                                     gender="F", birth_date="2001-08-27", birth_time="09:00:00"))
    body = {
        "user_id": "test",
        "email": "test",
        "password": "1234",
        "name": "test_record",
        "gender": "F",
        "birth_date": "2001-08-27",
        "birth_time": "09:00:00"
    }
    response = client.post("/members", json=body)

    assert response.status_code == 201
    assert response.json() == {
        "id": 2,
        "user_id": "test",
        "email": "test",
        "password": "1234",
        "name": "test_record",
        "gender": "F",
        "birth_date": "2001-08-27",
        "birth_time": "09:00:00"
    }
