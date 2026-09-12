"""add_country_to_job_offers

Revision ID: 29c892425359
Revises: 55d42b6b1ed3
Create Date: 2026-09-11 19:36:13.350291

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '29c892425359'
down_revision: Union[str, None] = '55d42b6b1ed3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add country column to job_offers
    op.add_column(
        "job_offers",
        sa.Column("country", sa.String(length=10), nullable=True),
    )

    # 2. Create index for fast filtering by country
    op.create_index(
        op.f("ix_job_offers_country"),
        "job_offers",
        ["country"],
        unique=False,
    )

    # 3. Backfill Computrabajo offers from source_url subdomain
    op.execute(
        """
        UPDATE job_offers
        SET country = substring(source_url from 'https://([a-z]{2})\\.computrabajo\\.com')
        WHERE portal = 'computrabajo' AND country IS NULL;
        """
    )

    # 4. Backfill GetOnBoard multi-country offers as 'latam'
    op.execute(
        """
        UPDATE job_offers
        SET country = 'latam'
        WHERE portal = 'getonboard' AND location LIKE '%,%' AND country IS NULL;
        """
    )

    # 5. Backfill GetOnBoard single-country offers
    op.execute(
        """
        UPDATE job_offers 
        SET country = CASE 
            WHEN location = 'Chile' THEN 'cl'
            WHEN location = 'Colombia' THEN 'co'
            WHEN location = 'Peru' THEN 'pe'
            WHEN location = 'Mexico' THEN 'mx'
            WHEN location = 'Argentina' THEN 'ar'
        END
        WHERE portal = 'getonboard' 
          AND country IS NULL 
          AND location IN ('Chile', 'Colombia', 'Peru', 'Mexico', 'Argentina');
        """
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_offers_country"), table_name="job_offers")
    op.drop_column("job_offers", "country")
