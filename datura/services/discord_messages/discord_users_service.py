from typing import List
from sqlalchemy import or_
from datura.services.discord_messages.database import Session, UserModel


class UserService:
    async def get_user(self, id: str) -> UserModel | None:
        """
        Retrieve a user model with its id

        Returns:
            UserModel | None: a full user model if found, or None
        """

        with Session() as session:
            user = session.query(UserModel).filter(UserModel.id == id).first()
            return user

    async def get_users(
        self,
        search: str,
        page: int = 1,
        limit: int = 10
    ) -> List[UserModel]:
        """
        Query users with search and pagination.

        Args:
            search (str): A search query to filter users by name and global name.
            page  (int): The page number for pagination.
            limit (int): The number of users per page.

        Returns:
            List[UserModel]: A list of UserModel objects representing users
            based on the search query and pagination.
        """

        with Session() as session:
            query = session.query(UserModel)
            if search:
                query = query.filter(or_(
                    UserModel.name.ilike(f"%{search}%"),
                    UserModel.global_name.ilike(f"%{search}%")
                ))

            users = query.limit(limit).offset((page - 1) * limit).all()
            return users
