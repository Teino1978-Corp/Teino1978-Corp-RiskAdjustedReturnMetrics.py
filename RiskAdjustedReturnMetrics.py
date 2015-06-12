import numpy
import pandas
import numpy.random as nrand


def vol(returns):
    return numpy.std(returns)


def beta(returns, market):
    m = numpy.matrix([returns, market])
    return numpy.cov(m)[0][1] / numpy.std(market)


def lpm(returns, threshold, order):
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    diff = threshold_array - returns
    diff = diff.clip(min=0)
    return numpy.sum(diff ** order)


def hpm(returns, threshold, order):
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    diff = returns - threshold_array
    diff = diff.clip(min=0)
    return numpy.sum(diff ** order)


def var(returns, alpha):
    sorted_returns = numpy.sort(returns)
    index = int(alpha * len(sorted_returns))
    return sorted_returns[index]


def cvar(returns, alpha):
    sorted_returns = numpy.sort(returns)
    index = int(alpha * len(sorted_returns))
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    return sum_var / index


def prices(returns, base):
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return numpy.array(s)


def dd(returns, tau):
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        # print(pos, pre, dd_i)
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    return drawdown


def max_dd(returns):
    max_drawdown = float('+inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i < max_drawdown:
            max_drawdown = drawdown_i
    return max_drawdown


if __name__ == "__main__":
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print(list(r))
    print(list(m))
    print(sorted(list(r)))
    print(sorted(list(m)))
    print(int(0.05 * 50))
    print(vol(r))
    print(beta(r, m))
    print(hpm(r, 0.0, 1))
    print(lpm(r, 0.0, 1))
    print(var(r, 0.05))
    print(cvar(r, 0.05))
    print(dd(r, 5))
    print(max_dd(r))