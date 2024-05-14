from enum import Enum, auto
import random
from datetime import datetime, timedelta
import pytz
from typing import Optional
import pydantic
from pydantic import BaseModel


class DateFilterType(Enum):
    PAST_24_HOURS = "PAST_24_HOURS"
    PAST_2_DAYS = "PAST_2_DAYS"
    PAST_WEEK = "PAST_WEEK"
    PAST_2_WEEKS = "PAST_2_WEEKS"
    PAST_MONTH = "PAST_MONTH"
    PAST_2_MONTH = "PAST_2_MONTH"
    PAST_YEAR = "PAST_YEAR"
    PAST_2_YEAR = "PAST_2_YEAR"


random_date_filters = [
    DateFilterType.PAST_24_HOURS,
    DateFilterType.PAST_24_HOURS,
    DateFilterType.PAST_24_HOURS,
    DateFilterType.PAST_24_HOURS,
    DateFilterType.PAST_24_HOURS,
    DateFilterType.PAST_2_DAYS,
    DateFilterType.PAST_2_DAYS,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_WEEK,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_2_WEEKS,
    DateFilterType.PAST_MONTH,
    DateFilterType.PAST_MONTH,
    DateFilterType.PAST_2_MONTH,
    DateFilterType.PAST_2_MONTH,
    DateFilterType.PAST_YEAR,
    DateFilterType.PAST_YEAR,
    DateFilterType.PAST_2_YEAR,
    DateFilterType.PAST_2_YEAR,
]


class DateFilter(BaseModel):
    start_date: Optional[datetime] = pydantic.Field(
        None,
        title="Start Date",
        description="The start date for the search query.",
    )

    end_date: Optional[datetime] = pydantic.Field(
        None,
        title="End Date",
        description="The end date for the search query.",
    )

    date_filter_type: Optional[DateFilterType] = pydantic.Field(
        None,
        title="Date filter enum",
        description="The date filter enum.",
    )


def get_specified_date_filter(date_filter: DateFilterType):
    now = datetime.now(pytz.utc).replace(second=0, microsecond=0)

    diff = timedelta(days=1)

    if date_filter == DateFilterType.PAST_24_HOURS:
        diff = timedelta(days=1)
    elif date_filter == DateFilterType.PAST_2_DAYS:
        diff = timedelta(days=2)
    elif date_filter == DateFilterType.PAST_WEEK:
        diff = timedelta(days=7)
    elif date_filter == DateFilterType.PAST_2_WEEKS:
        diff = timedelta(days=14)
    elif date_filter == DateFilterType.PAST_MONTH:
        diff = timedelta(days=30)
    elif date_filter == DateFilterType.PAST_2_MONTH:
        diff = timedelta(days=60)
    elif date_filter == DateFilterType.PAST_YEAR:
        diff = timedelta(days=365)
    elif date_filter == DateFilterType.PAST_2_YEAR:
        diff = timedelta(days=730)

    return DateFilter(
        start_date=now - diff,
        end_date=now,
        date_filter_type=date_filter,
    )


def get_random_date_filter():
    date_filter = random.choice(random_date_filters)
    return get_specified_date_filter(date_filter)
