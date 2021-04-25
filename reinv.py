from numpy import log, exp, sin, cos, sqrt, pi, power, outer, real, imag, arange, zeros, ones, conj, flipud, block, maximum
from numpy.fft import fft, ifft
from scipy.special import binom
from scipy.optimize import shgo, bisect 
from scipy.integrate import quad_vec

r, q, T, s0 = 0., 0., 10, 0.045

x0 = log(s0 / s0)
L = 10  
a = x0 - L * sqrt(T)
b = x0 + L * sqrt(T)

N = 2 ** 12 
u = arange(N).reshape(-1, 1) * pi / (b - a)

tau = range(5, T + 1)
M = len(tau)

def logger(*args):
    display = False
    if display:
        print(*args)

def cf_bs(dt, par):
    mu, sigma = par
    return lambda u: exp(1j * u * (r - q - 0.5 * sigma ** 2) * dt - 0.5 * sigma ** 2 * u ** 2 * dt)

def cf_enig(dt, par):
    alpha, beta, delta, mu = par
    sigma = 0
    return lambda u: exp(1j * u * (r - q - 0.5 * sigma ** 2 + delta * (sqrt(alpha ** 2 - (beta + 1) ** 2) - sqrt(alpha ** 2 - beta ** 2))) * dt - 0.5 * sigma ** 2 * u ** 2 * dt) * exp(delta * dt * (sqrt(alpha ** 2 - beta ** 2) - sqrt(alpha ** 2 - (beta + 1j * u) ** 2)))

def cf_evg(dt, par):
    c, sigma, theta, nu = par
    return lambda u: exp(1j * u * (r - q + 1 / nu * log(1 - theta * nu - 0.5 * sigma ** 2 * nu)) * dt - log(1 - 1j * theta * nu * u + 0.5 * sigma ** 2 * u ** 2  * nu) * dt / nu)

def get_xs(ti, V):
    _ = cf(tau[ti + 1] - tau[ti])(u) * V
    _[0] = 0.5 * _[0]
    
    t = tau[ti]

    def ct(x):
        return real(sum(exp(1j * outer(u, (x - a))) * _))[0]

    def gt(x):
        return maximum(1 - ((1 + s0 * exp(x)) / (1 + s0)) ** (T - t), 0) 

    def f(x):
        return gt(x) - ct(x)
    
    def fm(x):
        return -f(x)

    def df(x):
        tt = (T - t) * s0 / ((1 + s0) ** T) * ((1 + s0 * exp(x)) ** (T - t - 1)) * exp(x) if x <= 0 else 0 
        return tt - pi / (b - a) * imag(sum(exp(1j * outer(u, (x - a))) * arange(N).reshape(-1, 1) * _))[0]
    
    res = shgo(fm, bounds=([a, b],))
    if res['fun'] > 0:
        logger('t = ', t, 'c > g !')
        return False, 'cont'
    else:
        res1 = shgo(f, bounds=([a, b],))
        if res1['fun'] > 0:
            return False, 'ex'
        else:
            x, rr = bisect(f, res['x'], res1['x'], full_output=True)
            logger('t = ', t, 'finding root ...')
            logger('min c-g at ', res['x'], res['fun'])
            logger('max c-g at ', res1['x'], -res1['fun'])
            logger('got root at ', [x], f(x))
            logger(rr)
            logger()
            return True, x

#def G(x, ti): 
#    return 2 / (b - a) * quad_vec(lambda y: (1 - ((1 + s0 * exp(y)) / (1 + s0)) ** (T - tau[ti])) * cos(u * (y - a)), a, 0 if x >= 0 else x)[0]

def G(x, ti): # exact for integer p >= 1  
    x = 0 if x >= 0 else x
    p = T - tau[ti]
    V = zeros((N, 1))
    for k in arange(N): 
        if k: 
            res = 0
            for l in range(p + 1):
                res += binom(p, l) * (s0 ** l) * (l * exp(l * a) - exp(l * x) * (k * pi / (b - a) * sin(k * pi * (x - a) / (b - a)) + l * cos(k * pi * (x - a) / (b - a)))) / (k ** 2 * pi ** 2 + (b - a) ** 2 * l ** 2) 
            V[k] = 2 / (b - a) * ((b - a) ** 2 / ((1 + s0) ** p) * res + (b - a) * sin(k * pi * (x - a) / (b - a)) / (k * pi))
        else:
            res = a - x
            for l in range(1, p + 1):
                res += binom(p, l) * (s0 ** l) * (exp(l * a) - exp(l * x)) / l 
            V[k] = 2 / (b - a) * (1 / ((1 + s0) ** p) * res + (x - a))
    return V 

def C(x, ti, V):
    dt = tau[ti + 1] - tau[ti]
    v = pi / (b - a) * arange(1, N + 1).reshape(-1, 1)
    e1 = exp(1j * outer((x - a), v)).reshape(-1, 1)
    e2 = exp(1j * outer((b - a), v)).reshape(-1, 1)

    m = zeros((3 * N - 1, 1), dtype='complex_')                                        
    m[N - 1] = 1j * pi * (b - x) / (b - a)
    m[N: 2 * N] = (e2 - e1) / (v * (b - a) / pi)
    m[: N - 1] = - conj(flipud(m[N: 2 * N - 1]))
    m[2 * N: 3 * N - 1] = (e2[: N - 1] * e2[N - 1] - e1[: N - 1] * e1[N - 1]) / arange(N + 1, 2 * N).reshape(-1, 1)

    m_s = block([[m[N - 1: : -1]], [zeros((1, 1))], [m[2 * N - 2: N - 1: -1]]]).reshape(-1)
    m_c = m[3 * N - 2: N - 2: -1].reshape(-1)

    z = ones((2 * N, 1)).reshape(-1)
    z[2 * arange(1, N + 1) - 1] = -1

    uu = cf(dt)(u) * V
    uu[0] = 0.5 * uu[0]
    u_s = block([[uu], [zeros((N, 1))]]).reshape(-1) 
    fft_u_s = fft(u_s)
    xi_s = ifft(fft(m_s) * fft_u_s)
    xi_c = ifft(fft(m_c) * (z * fft_u_s))
    res = exp(-r * dt) / pi * imag(xi_s[:N] + flipud(xi_c[:N]))

    return res.reshape(-1, 1)

def value_cos():
    V = zeros((N, 1))               
    xs = zeros((M - 1, 1)).reshape(-1)
    V = G(b, M - 2) + C(b, M - 2, V)
    for i in arange(M - 3, -1, -1): 
        got, x_ = get_xs(i, V)   
        if got:
            xs[i] = x_
        else:
            xs[i] = a if x_ == 'cont' else b
        V = G(xs[i], i) + C(xs[i], i, V)

    _ = cf(tau[0])(u) * V 
    _[0] = 0.5 * _[0]
    return s0 * exp(xs), real(sum(exp(1j * outer(u, (x0 - a))) * _))[0]
