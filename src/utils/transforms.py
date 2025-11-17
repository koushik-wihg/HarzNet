import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class HFSE_REE_Ratios(BaseEstimator, TransformerMixin):
    # Create HFSE/REE ratio features (safe numeric coercion)
    def __init__(self, candidates=None):
        self.candidates = candidates or [
            ('Nb','Y'), ('Zr','Y'), ('Th','Yb'),
            ('Ce','Yb'), ('La','Ce'), ('Nb','La')
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        Xdf = X.copy()
        for num, den in self.candidates:
            col_num = next((c for c in Xdf.columns if c.lower() == num.lower()), None)
            col_den = next((c for c in Xdf.columns if c.lower() == den.lower()), None)

            if col_num and col_den:
                newname = f"{num}_{den}"
                vals = pd.to_numeric(Xdf[col_num], errors="coerce")
                vals2 = pd.to_numeric(Xdf[col_den], errors="coerce")
                ratio = vals / (vals2.replace({0: np.nan}))
                Xdf[newname] = ratio.fillna(0.0)

        return Xdf


class PivotILRTransformer(BaseEstimator, TransformerMixin):
    # ILR transformer without external dependencies
    def __init__(self, comp_cols=(), zero_replace_factor=1e-6):
        self.comp_cols = tuple(comp_cols)
        self.zero_replace_factor = zero_replace_factor

    def fit(self, X, y=None):
        self.input_columns_ = list(X.columns)
        self.comp_idx_ = [self.input_columns_.index(c) for c in self.comp_cols]

        comp_vals = X.iloc[:, self.comp_idx_].to_numpy(dtype=float)
        pos = comp_vals[(~np.isnan(comp_vals)) & (comp_vals > 0)]
        self.eps_ = (pos.min() * self.zero_replace_factor) if pos.size > 0 else self.zero_replace_factor

        self.noncomp_cols_ = [
            c for i, c in enumerate(self.input_columns_) if i not in self.comp_idx_
        ]
        return self

    def _close(self, A):
        s = A.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return A / s

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_df = pd.DataFrame(X, columns=self.input_columns_)
        else:
            X_df = X

        X_df = pd.DataFrame(X_df, columns=self.input_columns_, index=X_df.index)

        comps = X_df.loc[:, list(self.comp_cols)].to_numpy(dtype=float)
        comps = np.where(np.isnan(comps), self.eps_, comps)
        comps = np.where(comps <= 0, self.eps_, comps)

        Xc = self._close(comps)
        n, k = Xc.shape

        if k < 2:
            raise ValueError("Need at least 2 compositional parts for ILR")

        ilr = np.zeros((n, k - 1))
        for j in range(k - 1):
            gm = np.exp(np.mean(np.log(Xc[:, j+1:]), axis=1))
            scale = math.sqrt((k - j - 1) / (k - j))
            ilr[:, j] = scale * np.log(Xc[:, j] / gm)

        ilr_cols = [f"ilr_{c}_vs_rest" for c in self.comp_cols[:-1]]
        ilr_df = pd.DataFrame(ilr, columns=ilr_cols, index=X_df.index)

        if len(self.noncomp_cols_) > 0:
            noncomp_df = X_df[self.noncomp_cols_]
            return pd.concat([ilr_df, noncomp_df], axis=1)
        return ilr_df

# ---- compositional CLR transform ----
def clr_transform(X):
    """Center log-ratio transform for a 2D numeric array-like.
    Accepts numpy arrays or pandas DataFrames (rows = samples, cols = comps),
    adds a tiny offset to avoid log(0).
    """
    import numpy as _np
    X = _np.asarray(X, dtype=float)
    offset = 1e-9
    X = X + offset
    logX = _np.log(X)
    gm = _np.mean(logX, axis=1, keepdims=True)
    clr = logX - gm
    return clr

# END of CLR
