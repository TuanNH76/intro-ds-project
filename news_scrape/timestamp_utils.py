from datetime import datetime, timedelta, timezone


class Timestamp:
    """
    Class for retrieving fixed timestamps relative to the current time
    """

    @staticmethod
    def start_of_day() -> datetime:
        """
        Retrieve the start of current day - 0h00 of the current day
        Output:
            0h00 of the current day in datetime utc type
        """
        now = datetime.now(timezone.utc)
        start_of_day = datetime(
            now.year, now.month, now.day, 0, 0, 0, tzinfo=timezone.utc
        )
        return start_of_day

    @staticmethod
    def start_of_week() -> datetime:
        """
        Retrieve the start of current week - 0h00 of the current week
        Output:
            0h00 of the current week in datetime utc type
        """
        now = datetime.now(timezone.utc)
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        return start_of_week

    @staticmethod
    def past_noon_or_midnight() -> datetime:
        """
        Retrieve the past noon or midnight of the current day, depends on if the current time is past noon or midnight
        Output:
            past noon or midnight of the current day in datetime utc type
        """
        now = datetime.now(timezone.utc)
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        noon = now.replace(hour=12, minute=0, second=0, microsecond=0)
        if now < noon:
            nearest_time = midnight  # 12 AM is the nearest past time
            return nearest_time
        else:
            nearest_time = noon  # 12 PM is the nearest past time
            return nearest_time

    @staticmethod
    def past_3hour_timestamp() -> datetime:
        now = datetime.now(timezone.utc)

        nearest_hour = (now.hour // 3) * 3

        nearest_time = now.replace(hour=nearest_hour, minute=0, second=0, microsecond=0)

        return nearest_time

    @staticmethod
    def now() -> datetime:
        """
        Retrieve the current time in datetime utc type
        Output:
            current time in datetime utc type
        """
        now = datetime.now(timezone.utc)
        return now


class TimestampClient:
    def __init__(self):
        self.past_noon_or_midnight = Timestamp.past_noon_or_midnight()
        self.start_of_day = Timestamp.start_of_day()
        self.start_of_week = Timestamp.start_of_week()
        self.past_3hour_timestamp = Timestamp.past_3hour_timestamp()
        self.now = Timestamp.now()