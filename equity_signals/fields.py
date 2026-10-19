"""Input fields of the signal functions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class Fields:
    """Daily fields of a set of stocks, one frame per field.

    Every frame is indexed by date in ascending order and has one column
    per stock. Missing values are NaN.

    Attributes:
        close: Closing price.
        high: Highest price of the day.
        low: Lowest price of the day.
        volume: Number of shares traded.
        market_cap: Market value of the equity at the close.
        rating: Consensus analyst rating, or None when not available.
    """

    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    market_cap: pd.DataFrame
    rating: Optional[pd.DataFrame] = None

    def aligned(self) -> "Fields":
        """Returns a copy with every frame on the index and columns of close.

        A missing rating frame becomes a frame of NaN.

        Returns:
            Fields with frames of float on a common index and columns.
        """
        index, columns = self.close.index, self.close.columns

        def fit(frame: pd.DataFrame) -> pd.DataFrame:
            """Puts a frame on the index and columns of close as floats.

            Args:
                frame: Any frame of the fields.

            Returns:
                Frame of floats on the common index and columns.
            """
            return frame.reindex(index=index, columns=columns).astype(float)

        rating = self.rating
        if rating is None:
            rating = pd.DataFrame(np.nan, index=index, columns=columns)
        return Fields(close=fit(self.close), high=fit(self.high),
                      low=fit(self.low), volume=fit(self.volume),
                      market_cap=fit(self.market_cap), rating=fit(rating))

    def rows(self, keep: pd.Series) -> "Fields":
        """Returns a copy restricted to the rows where keep is True.

        Args:
            keep: Boolean Series on the index of the frames.

        Returns:
            Fields with the selected rows of every frame.
        """
        mask = np.asarray(keep, dtype=bool)

        def cut(frame: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            """Keeps the selected rows of a frame.

            Args:
                frame: Any frame of the fields, or None.

            Returns:
                The frame with the selected rows, or None for None.
            """
            return None if frame is None else frame[mask]

        return Fields(close=cut(self.close), high=cut(self.high),
                      low=cut(self.low), volume=cut(self.volume),
                      market_cap=cut(self.market_cap),
                      rating=cut(self.rating))
