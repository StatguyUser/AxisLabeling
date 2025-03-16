"""
Created on Sat Mar 15 15:56:04 2025
@author: statguyuser

A Python implementation of several axis‐labeling algorithms originally implemented in the R “labeling” package
by Justin Talbot et al. The functions include Heckbert’s, Wilkinson’s, Extended‐Wilkinson, Nelder’s, R’s pretty,
Matplotlib’s, gnuplot’s, Sparks’, and Thayer & Storer’s algorithms.

Each function expects the data range endpoints (dmin, dmax) and a target number of labels (m). Many functions
are written in an “optimization search” style similar to the R source.

References:
 - Heckbert, P. S. (1990) Nice numbers for graph labels, Graphics Gems I.
 - Wilkinson, L. (2005) The Grammar of Graphics.
 - Talbot, J., Lin, S., Hanrahan, P. (2010) An Extension of Wilkinson’s Algorithm for Positioning Tick Labels on Axes.
"""

import math
import numpy as np

class Labeler:
    """
    A Python implementation of several axis‐labeling algorithms originally implemented in the R “labeling” package
    by Justin Talbot et al. The methods include Heckbert’s, Wilkinson’s, Extended‐Wilkinson, Nelder’s, R’s pretty,
    Matplotlib’s, gnuplot’s, Sparks’, and Thayer & Storer’s algorithms.
    
    The common parameters:
      - dmin: minimum data value.
      - dmax: maximum data value.
      - m: desired number of labels.
    """
    def __init__(self, dmin, dmax, m):
        self.dmin = dmin
        self.dmax = dmax
        self.m = m

    # -----------------------
    # Helper Methods
    # -----------------------
    def _heckbert_nicenum(self, x, do_round):
        """Helper for Heckbert’s algorithm to choose a “nice number”."""
        if x <= 0:
            return 0
        e = math.floor(math.log10(x))
        f = x / (10 ** e)
        if do_round:
            if f < 1.5:
                nf = 1
            elif f < 3:
                nf = 2
            elif f < 7:
                nf = 5
            else:
                nf = 10
        else:
            if f <= 1:
                nf = 1
            elif f <= 2:
                nf = 2
            elif f <= 5:
                nf = 5
            else:
                nf = 10
        return nf * (10 ** e)
    
    def _wilkinson_nice_scale(self, dmin, dmax, k, Q, mincoverage, m):
        """
        Helper for Wilkinson’s algorithm: searches for a “nice” scale given k ticks.
        Returns a dict with keys 'lmin', 'lmax', 'lstep', 'score', or None.
        """
        Q2 = [10] + Q  # prepend 10 as in the R code
        range_val = dmax - dmin
        intervals = k - 1
        granularity = 1 - abs(k - m) / m
        delta = range_val / intervals
        base = math.floor(math.log10(delta)) if delta > 0 else 0
        dbase = 10 ** base
        best = None
        nQ = len(Q2)
        for i, q in enumerate(Q2):
            tdelta = q * dbase
            tmin = math.floor(dmin / tdelta) * tdelta
            tmax = tmin + intervals * tdelta
            if tmin <= dmin and tmax >= dmax:
                adjust = 1 if (tmin <= 0 and tmax >= 0) else 0
                roundness = 1 - (i - adjust) / nQ
                coverage = (dmax - dmin) / (tmax - tmin)
                if coverage > mincoverage:
                    tnice = granularity + roundness + coverage
                    if (best is None) or (tnice > best['score']):
                        best = {'lmin': tmin, 'lmax': tmax, 'lstep': tdelta, 'score': tnice}
        return best

    def _simplicity(self, q, Q, j, lmin, lmax, lstep):
        """Simplicity score for Extended Wilkinson’s algorithm."""
        eps = np.finfo(float).eps * 100
        n = len(Q)
        try:
            i = Q.index(q)
        except ValueError:
            i = 0
        v = 1 if (abs(lmin % lstep) < eps or abs(lstep - (lmin % lstep)) < eps) and (lmin <= 0 and lmax >= 0) else 0
        return 1 - (i) / (n - 1) - j + v

    def _simplicity_max(self, q, Q, j):
        """Maximum possible simplicity score."""
        n = len(Q)
        try:
            i = Q.index(q)
        except ValueError:
            i = 0
        v = 1
        return 1 - (i) / (n - 1) - j + v

    def _coverage(self, dmin, dmax, lmin, lmax):
        """Coverage score."""
        range_val = dmax - dmin
        return 1 - 0.5 * (((dmax - lmax) ** 2 + (dmin - lmin) ** 2) / ((0.1 * range_val) ** 2))

    def _coverage_max(self, dmin, dmax, span):
        """Maximum coverage score."""
        range_val = dmax - dmin
        if span > range_val:
            half = (span - range_val) / 2
            return 1 - 0.5 * ((half ** 2 + half ** 2) / ((0.1 * range_val) ** 2))
        else:
            return 1

    def _density(self, k, m, dmin, dmax, lmin, lmax):
        """Density score."""
        r = (k - 1) / (lmax - lmin)
        rt = (m - 1) / (max(lmax, dmax) - min(dmin, lmin))
        return 2 - max(r / rt, rt / r)

    def _density_max(self, k, m):
        """Maximum density score."""
        if k >= m:
            return 2 - (k - 1) / (m - 1)
        else:
            return 1

    def _legibility(self, lmin, lmax, lstep):
        """Legibility score (here always 1 as in the R implementation)."""
        return 1

    def _matplotlib_scale_range(self, vmin, vmax, bins):
        """Helper for matplotlib_labeling: computes a scale and offset."""
        threshold = 100
        dv = abs(vmax - vmin)
        maxabsv = max(abs(vmin), abs(vmax))
        if maxabsv == 0 or dv / maxabsv < 1e-12:
            return (1, 0)
        meanv = 0.5 * (vmin + vmax)
        if abs(meanv) / dv < threshold:
            offset = 0
        elif meanv > 0:
            offset = 10 ** math.floor(math.log10(meanv))
        else:
            offset = -10 ** math.floor(math.log10(-meanv))
        exp = math.floor(math.log10(dv / bins))
        scale = 10 ** exp
        return (scale, offset)

    # -----------------------
    # Public Labeling Methods
    # -----------------------
    def heckbert(self):
        """
        Returns a vector of axis label locations using Heckbert's nice number algorithm.
        """
        rng = self._heckbert_nicenum(self.dmax - self.dmin, False)
        lstep = self._heckbert_nicenum(rng / (self.m - 1), True)
        lmin = math.floor(self.dmin / lstep) * lstep
        lmax = math.ceil(self.dmax / lstep) * lstep
        return np.arange(lmin, lmax + lstep/2, lstep)

    def wilkinson(self, Q=[1, 5, 2, 2.5, 3, 4, 1.5, 7, 6, 8, 9], mincoverage=0.8):
        """
        Returns axis label locations using Wilkinson's labeling algorithm.
        """
        best = None
        lower = max(math.floor(self.m / 2), 2)
        upper = math.ceil(6 * self.m)
        for k in range(lower, upper + 1):
            result = self._wilkinson_nice_scale(self.dmin, self.dmax, k, Q, mincoverage, self.m)
            if result is not None:
                if (best is None) or (result['score'] > best['score']):
                    best = result
        if best is None:
            return np.linspace(self.dmin, self.dmax, self.m)
        return np.arange(best['lmin'], best['lmax'] + best['lstep']/2, best['lstep'])

    def extended(self, Q=[1, 5, 2, 2.5, 4, 3], only_loose=False, w=[0.25, 0.2, 0.5, 0.05]):
        """
        Extended Wilkinson’s algorithm for positioning tick labels.
        """
        eps = np.finfo(float).eps * 100
        if self.dmin > self.dmax:
            self.dmin, self.dmax = self.dmax, self.dmin
        if (self.dmax - self.dmin) < eps or (self.dmax - self.dmin) > math.sqrt(np.finfo(float).max):
            return np.linspace(self.dmin, self.dmax, self.m)
        
        best = {'score': -2}
        j = 1
        while True:
            stop_j = False
            for q in Q:
                sm = self._simplicity_max(q, Q, j)
                if (w[0] * sm + w[1] + w[2] + w[3]) < best['score']:
                    stop_j = True
                    break
                k = 2
                while True:
                    dm = self._density_max(k, self.m)
                    if (w[0] * sm + w[1] + w[2] * dm + w[3]) < best['score']:
                        break
                    delta = (self.dmax - self.dmin) / ((k + 1) * j * q)
                    z = 0 if delta <= 0 else math.ceil(math.log10(delta))
                    while True:
                        step = j * q * (10 ** z)
                        cm = self._coverage_max(self.dmin, self.dmax, step * (k - 1))
                        if (w[0] * sm + w[1] * cm + w[2] * dm + w[3]) < best['score']:
                            break
                        min_start = math.floor(self.dmax / step) * j - (k - 1) * j
                        max_start = math.ceil(self.dmin / step) * j
                        if min_start > max_start:
                            z += 1
                            continue
                        for start in range(int(min_start), int(max_start) + 1):
                            lmin_val = start * (step / j)
                            lmax_val = lmin_val + step * (k - 1)
                            lstep = step
                            s_val = self._simplicity(q, Q, j, lmin_val, lmax_val, lstep)
                            c_val = self._coverage(self.dmin, self.dmax, lmin_val, lmax_val)
                            g_val = self._density(k, self.m, self.dmin, self.dmax, lmin_val, lmax_val)
                            l_val = self._legibility(lmin_val, lmax_val, lstep)
                            score = w[0] * s_val + w[1] * c_val + w[2] * g_val + w[3] * l_val
                            if (score > best['score'] and
                                (not only_loose or (lmin_val <= self.dmin and lmax_val >= self.dmax))):
                                best = {'lmin': lmin_val, 'lmax': lmax_val, 'lstep': lstep, 'score': score}
                        z += 1
                        if (10 ** z) * j * q > (self.dmax - self.dmin) * 10:
                            break
                    k += 1
                    if k > self.m * 10:
                        break
            if stop_j:
                break
            j += 1
            if j > self.m * 10:
                break
        if 'lmin' not in best:
            return np.linspace(self.dmin, self.dmax, self.m)
        return np.arange(best['lmin'], best['lmax'] + best['lstep']/2, best['lstep'])

    def nelder(self, Q=[1, 1.2, 1.6, 2, 2.5, 3, 4, 5, 6, 8, 10]):
        """
        Nelder’s labeling algorithm.
        """
        ntick = math.floor(self.m)
        tol = 5e-6
        bias = 1e-4
        intervals = self.m - 1
        x = abs(self.dmax)
        if x == 0:
            x = 1
        if not ((self.dmax - self.dmin) / x > tol):
            return np.linspace(self.dmin, self.dmax, ntick)
        step = (self.dmax - self.dmin) / intervals
        s_val = step
        while s_val <= 1:
            s_val *= 10
        while s_val > 10:
            s_val /= 10
        x_val = s_val - bias
        unit_index = 0
        for i, val in enumerate(Q):
            if x_val < val:
                unit_index = i
                break
        if unit_index >= len(Q):
            unit_index = len(Q) - 1
        step = step * Q[unit_index] / s_val
        range_val = step * intervals
        x_val = 0.5 * (1 + (self.dmin + self.dmax - range_val) / step)
        j = math.floor(x_val - bias)
        valmin = step * j
        if self.dmin > 0 and range_val >= self.dmax:
            valmin = 0
        valmax = valmin + range_val
        if not (self.dmax > 0 or range_val < -self.dmin):
            valmax = 0
            valmin = -range_val
        return np.arange(valmin, valmax + step/2, step)

    def rpretty(self, m=None, n=None, min_n=None, shrink_sml=0.75, high_u_bias=1.5, u5_bias=None):
        """
        R's pretty algorithm implemented in Python.
        """
        if m is None:
            m = self.m
        if n is None:
            n = math.floor(m) - 1
        if min_n is None:
            min_n = n // 3
        ndiv = n
        h = high_u_bias
        if u5_bias is None:
            u5_bias = 0.5 + 1.5 * high_u_bias
        dx = self.dmax - self.dmin
        if dx == 0 and self.dmax == 0:
            cell = 1
            i_small = True
            U = 1
        else:
            cell = max(abs(self.dmin), abs(self.dmax))
            U = 1 + (1 / (1 + h) if u5_bias >= 1.5 * h + 0.5 else 1.5 / (1 + u5_bias))
            i_small = dx < (cell * U * max(1, ndiv) * 1e-07 * 3)
        if i_small:
            if cell > 10:
                cell = 9 + cell / 10
            cell = cell * shrink_sml
            if min_n > 1:
                cell = cell / min_n
        else:
            cell = dx
            if ndiv > 1:
                cell = cell / ndiv
        if cell < 20 * 1e-07:
            cell = 20 * 1e-07
        base = 10 ** math.floor(math.log10(cell))
        unit = base
        if (2 * base - cell) < h * (cell - unit):
            unit = 2 * base
            if (5 * base - cell) < u5_bias * (cell - unit):
                unit = 5 * base
                if (10 * base - cell) < h * (cell - unit):
                    unit = 10 * base
        ns = math.floor(self.dmin / unit + 1e-07)
        nu = math.ceil(self.dmax / unit - 1e-07)
        while ns * unit > self.dmin + (1e-07 * unit):
            ns -= 1
        while nu * unit < self.dmax - (1e-07 * unit):
            nu += 1
        k = math.floor(0.5 + (nu - ns))
        if k < min_n:
            diff = min_n - k
            if ns >= 0:
                nu = nu + diff / 2
                ns = ns - diff / 2 + (diff % 2)
            else:
                ns = ns - diff / 2
                nu = nu + diff / 2 + (diff % 2)
            ndiv = min_n
        else:
            ndiv = k
        graphmin = ns * unit
        graphmax = nu * unit
        return np.arange(graphmin, graphmax + unit/2, unit)

    def matplotlib_labeling(self):
        """
        Matplotlib’s labeling algorithm.
        """
        steps = [1, 2, 5, 10]
        nbins = self.m
        trim = True
        vmin = self.dmin
        vmax = self.dmax
        scale, offset = self._matplotlib_scale_range(vmin, vmax, nbins)
        vmin = vmin - offset
        vmax = vmax - offset
        rawStep = (vmax - vmin) / nbins
        scaledRawStep = rawStep / scale
        bestMax = vmax
        bestMin = vmin
        scaledStep = 1
        for step in steps:
            if step >= scaledRawStep:
                scaledStep = step * scale
                bestMin = scaledStep * math.floor(vmin / scaledStep)
                bestMax = bestMin + scaledStep * nbins
                if bestMax >= vmax:
                    break
        if trim:
            extraBins = math.floor((bestMax - vmax) / scaledStep)
            nbins = nbins - extraBins
        graphMin = bestMin + offset
        graphMax = graphMin + nbins * scaledStep
        return np.arange(graphMin, graphMax + scaledStep/2, scaledStep)

    def gnuplot_labeling(self):
        """
        gnuplot’s labeling algorithm.
        """
        ntick = math.floor(self.m)
        power = 10 ** math.floor(math.log10(self.dmax - self.dmin))
        norm_range = (self.dmax - self.dmin) / power
        p = (ntick - 1) / norm_range
        if p > 40:
            t = 0.05
        elif p > 20:
            t = 0.1
        elif p > 10:
            t = 0.2
        elif p > 4:
            t = 0.5
        elif p > 2:
            t = 1
        elif p > 0.5:
            t = 2
        else:
            t = math.ceil(norm_range)
        d_val = t * power
        graphmin = math.floor(self.dmin / d_val) * d_val
        graphmax = math.ceil(self.dmax / d_val) * d_val
        return np.arange(graphmin, graphmax + d_val/2, d_val)

    def sparks(self):
        """
        Sparks' labeling algorithm translated from R.
        """
        fm = self.m - 1
        ratio = 0.0
        key = 1
        kount = 0
        r = self.dmax - self.dmin
        b = self.dmin

        while ratio <= 0.8:
            while key <= 2:
                while r <= 1:
                    kount += 1
                    r *= 10
                while r > 10:
                    kount -= 1
                    r /= 10
                b = b * (10 ** kount)
                if b < 0 and b != math.trunc(b):
                    b = b - 1
                b = math.trunc(b) / (10 ** kount)
                r = (self.dmax - b) / fm
                kount = 0
                key += 2
            fstep = math.trunc(r)
            if fstep != r:
                fstep = fstep + 1
            if r < 1.5:
                fstep = fstep - 0.5
            fstep = fstep / (10 ** kount)
            ratio = (self.dmax - self.dmin) * (fm * fstep)
            kount = 1
            key = 2

        fmin = b
        c = fstep * math.trunc(b / fstep)
        if c < 0 and c != b:
            c = c - fstep
        if (c + fm * fstep) > self.dmax:
            fmin = c
        labels = np.arange(fmin, fstep * (self.m - 1) + fstep / 10, fstep)
        return labels

    def thayer(self):
        """
        Thayer and Storer’s labeling algorithm.
        """
        r = self.dmax - self.dmin
        b = self.dmin
        kount = 0
        kod = 0
        while kod < 2:
            while r <= 1:
                kount += 1
                r *= 10
            while r > 10:
                kount -= 1
                r /= 10
            b = b * (10 ** kount)
            if b < 0:
                b = b - 1
            ib = math.trunc(b)
            b = ib
            b = b / (10 ** kount)
            r = self.dmax - b
            a = r / (self.m - 1)
            kount = 0
            while a <= 1:
                kount += 1
                a *= 10
            while a > 10:
                kount -= 1
                a /= 10
            ia = math.trunc(a)
            if ia == 6:
                ia = 7
            if ia == 8:
                ia = 9
            aa = -0.5 if a < 1.5 else 0
            a = (aa + 1 + ia) / (10 ** kount)
            test = (self.m - 1) * a
            test1 = (self.dmax - self.dmin) / test
            if test1 > 0.8:
                kod = 2
            if kod < 2:
                kount = 1
                r = self.dmax - self.dmin
                b = self.dmin
                kod += 1
        iab = math.trunc(b / a)
        if iab < 0:
            iab = iab - 1
        c = a * iab
        d_val = c + (self.m - 1) * a
        if d_val >= self.dmax:
            b = c
        valmin = b
        valmax = b + a * (self.m - 1)
        return np.arange(valmin, valmax + a/2, a)


# -----------------------
# Example usage
# -----------------------
if __name__ == '__main__':
    dmin, dmax, m = 7.1, 14.1, 4
    labeler = Labeler(dmin, dmax, m)
    print("Heckbert labels:", labeler.heckbert())
    print("Wilkinson labels:", labeler.wilkinson())
    print("Extended labels:", labeler.extended())
    print("Nelder labels:", labeler.nelder())
    print("rpretty labels:", labeler.rpretty())
    print("Matplotlib labels:", labeler.matplotlib_labeling())
    print("gnuplot labels:", labeler.gnuplot_labeling())
    print("Sparks labels:", labeler.sparks())
    print("Thayer labels:", labeler.thayer())


