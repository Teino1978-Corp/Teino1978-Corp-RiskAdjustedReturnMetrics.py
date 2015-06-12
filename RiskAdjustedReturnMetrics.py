import numpy
import numpy.random as nrand

"""
Note - for some of the metrics the absolute value is returns. This is because if the risk (loss) is higher we want to
discount the expected excess return from the portfolio by a higher amount. Therefore risk should be positive.
"""


def vol(returns):
    # Return the standard deviation of returns
    return numpy.std(returns)


def beta(returns, market):
    # Create a matrix of [returns, market]
    m = numpy.matrix([returns, market])
    # Return the covariance of m divided by the standard deviation of the market returns
    return numpy.cov(m)[0][1] / numpy.std(market)


def lpm(returns, threshold, order):
    # This method returns a lower partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the threshold and the returns
    diff = threshold_array - returns
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)


def hpm(returns, threshold, order):
    # This method returns a higher partial moment of the returns
    # Create an array he same length as returns containing the minimum return threshold
    threshold_array = numpy.empty(len(returns))
    threshold_array.fill(threshold)
    # Calculate the difference between the returns and the threshold
    diff = returns - threshold_array
    # Set the minimum of each to 0
    diff = diff.clip(min=0)
    # Return the sum of the different to the power of order
    return numpy.sum(diff ** order) / len(returns)


def var(returns, alpha):
    # This method calculates the historical simulation var of the returns
    sorted_returns = numpy.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # VaR should be positive
    return abs(sorted_returns[index])


def cvar(returns, alpha):
    # This method calculates the condition VaR of the returns
    sorted_returns = numpy.sort(returns)
    # Calculate the index associated with alpha
    index = int(alpha * len(sorted_returns))
    # Calculate the total VaR beyond alpha
    sum_var = sorted_returns[0]
    for i in range(1, index):
        sum_var += sorted_returns[i]
    # Return the average VaR
    # CVaR should be positive
    return abs(sum_var / index)


def prices(returns, base):
    # Converts returns into prices
    s = [base]
    for i in range(len(returns)):
        s.append(base * (1 + returns[i]))
    return numpy.array(s)


def dd(returns, tau):
    # Returns the draw-down given time period tau
    values = prices(returns, 100)
    pos = len(values) - 1
    pre = pos - tau
    drawdown = float('+inf')
    # Find the maximum drawdown given tau
    while pre >= 0:
        dd_i = (values[pos] / values[pre]) - 1
        if dd_i < drawdown:
            drawdown = dd_i
        pos, pre = pos - 1, pre - 1
    # Drawdown should be positive
    return abs(drawdown)


def max_dd(returns):
    # Returns the maximum draw-down for any tau in (0, T) where T is the length of the return series
    max_drawdown = float('-inf')
    for i in range(0, len(returns)):
        drawdown_i = dd(returns, i)
        if drawdown_i > max_drawdown:
            max_drawdown = drawdown_i
    # Max draw-down should be positive
    return abs(max_drawdown)


def test():
    # This is just a testing method
    r = nrand.uniform(-1, 1, 50)
    m = nrand.uniform(-1, 1, 50)
    print("vol =", vol(r))
    print("beta =", beta(r, m))
    print("hpm(0.0)_1 =", hpm(r, 0.0, 1))
    print("lpm(0.0)_1 =", lpm(r, 0.0, 1))
    print("VaR(0.05) =", var(r, 0.05))
    print("CVaR(0.05) =", cvar(r, 0.05))
    print("Drawdown(5) =", dd(r, 5))
    print("Max Drawdown =", max_dd(r))


def test_two():
    print("Vol", "Beta", "HPM", "LPM", "VaR", "CVaR", "DD", "MDD")
    for i in range(1000):
        r = nrand.uniform(-1, 1, 50)
        m = nrand.uniform(-1, 1, 50)
        print(vol(r), beta(r, m), hpm(r, 0.0, 1), lpm(r, 0.0, 1), var(r, 0.05), cvar(r, 0.05), dd(r, 5), max_dd(r))


if __name__ == "__main__":
    test_two()