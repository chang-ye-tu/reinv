import os
os.chdir(os.path.dirname(__file__))
import codecs
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import pandas as pd
from datetime import datetime
from tabulate import tabulate
import seaborn as sns
from numpy import shape, log, exp, pi, diff, mean, array, sqrt, inf, linspace, append
from scipy.special import gamma, kv
from scipy.optimize import fmin, shgo

# render python tabulate latex_raw into latex_booktabs
def latex_booktabs(s):
    l = s.split('\\hline\n')
    return l[0] + '\\toprule\n' + '\\midrule\n'.join(l[1:-1]) + '\\bottomrule\n' + l[-1]

#####################################################################
#
# Collect DGS30 & IRR Data and Processing
#
#####################################################################

# Download bond_publish.csv from https://www.tpex.org.tw/web/bond/publish/international_bond_search/memo_org.php?l=zh-tw
# 債券代號, 債券簡稱, 發行人, 發行日期, 到期日期, 年期, 計價幣別, 發行總額, 票面利率, 隱含利率, 提前贖回權, 贖回型態, 承銷商或財務顧問, 流動量提供者, 國際編碼, 發行資料, 是否具損失吸收能力
# Bond Code, Short Name, Issuer, Issuing Date, Maturity Date, Tenor, Currency Denomination, Amount of Issuance, Coupon(%), IRR(%), Early Redemption, Non-Call Period(Year) x Call Frequency(Year), Securities Underwriter or Financial Consulting Company, Liquidity Provider, ISIN, Bond Database, TLAC/bail-in eligible

def trans(n):
    # transform % into actual numerical value
    b_trans = True
    return 0.01 * n if b_trans else n 

slist, tmp, data = [], [], []
for ii, l in enumerate(codecs.open('bond_publish.csv', 'r', 'big5')):
    if ii >= 4:
        s = l.strip()
        if s.find('"F') == 0:
            tmp = [s + ' ']
        else:
            tmp.append(s)
            slist.append(''.join(tmp))
            tmp = []
tmp = []
for sl in slist:
    l = sl.split('","')
    l[0] = l[0][1:]     # remove leading "
    l[-1] = l[-1][:-1]  # remove trailing "
    tmp.append(l)

for l in tmp:
    name = l[2].lstrip()
    d_issue = datetime.strptime(l[3], '%Y/%m/%d')
    tenor = float(l[5])
    coupon = trans(float(l[8]))
    irr = trans(float(l[9]))
    if l[10] == 'Applicable':
        _ = l[11]
        if _.find(' x ') > 0:
            a, b = _.split(' x ')
            non_call, freq = int(float(a)), int(float(b))
            data.append((d_issue, name, tenor, coupon, irr, non_call, freq))          
dfi = pd.DataFrame(data=sorted(data), columns=['date', 'name', 'tenor', 'coupon', 'irr', 'non_call', 'freq'])
dfi = dfi[(dfi['coupon'] <= 1e-10) & (dfi['irr'] > 0) & (dfi['tenor'] >= 30) & (dfi['non_call'] == 5)]
dfi_ = dfi[['date', 'irr']].copy().groupby('date')
dfi = dfi_.mean()
dfi.reset_index(level=dfi.index.names, inplace=True)
span = dfi['date'].max() - dfi['date'].min()
dfif = dfi['date'].diff().dropna()

df_ = dfif.apply(lambda x: x.total_seconds() / 86400.).to_frame('days')
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
sns.histplot(data=df_, x='days', kde=True, bins=50, stat='count', legend=False)
textstr = '\n'.join((r'$\mathrm{mean}=%.6f$' % (df_.mean(),), r'$\mathrm{median}=%.6f$' % (df_.median(),), r'$\mathrm{std}=%.6f$' % (df_.std(),)))
ax.text(0.5, 0.5, textstr, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.2))
fig.tight_layout()
fig.savefig('days.pdf', bbox_inches='tight')

Dt = span.total_seconds() / 86400 / 365 / dfi['date'].count()
open('irr.txt', 'w').write('\n'.join(['data', str(Dt)] + [str(i) for i in dfi['irr']]))

# Download DGS30.csv from https://fred.stlouisfed.org/series/DGS30
data = []
for ii, sl in enumerate(open('DGS30.csv')):
    if ii:
        l = sl.strip().split(',')
        try:
            d_date = datetime.strptime(l[0], '%Y-%m-%d')
            value = trans(float(l[1]))
            data.append((d_date, value))
        except:
            pass

fig, ax = plt.subplots(2, 1, figsize=(9, 6))

dfd = pd.DataFrame(data=data, columns=['date', 'dgs30'])
dfd[dfd['date'] >= '2006-01-02'].plot(x='date', y='dgs30', figsize=(9, 6), grid=True, ax=ax[0])
ax[0].set_xlabel('')
ax[0].set_ylabel('Rate')

data = []
offset = 5
for _, row in dfi.iterrows():
    item = dfd.iloc[dfd[dfd['date'] == row['date']].index.item() - offset]
    data.append([row['date'], row['irr'], item['date'], item['dgs30']])
df = pd.DataFrame(data=data, columns=['date', 'irr', 'date_dgs30', 'dgs30'])
#df = pd.merge(dfi, dfd, on='date')

df.plot(x='date', y=['dgs30', 'irr'], figsize=(9, 6), grid=True, style=['+-','o-','.--','s:'], ax=ax[1])
ax[1].set_xlabel('Date')
ax[1].set_ylabel('Rate')
fig.tight_layout()
fig.savefig('irr_dgs30.pdf', bbox_inches='tight')

df0 = df.copy(deep=True)
df0['diff'] = df0['irr'] - df0['dgs30']
df0['ratio'] = (df0['diff']) / df0['dgs30']
#print(df0.iloc[df0['diff'].idxmin()])
#open('df0.txt', 'w').write('\n'.join(['data'] + [str(i) for i in dfi['irr']]))

from statsmodels.tsa.stattools import adfuller
ll = []
for ii, series in enumerate((df0['diff'], df0['diff'].diff().dropna())):
    r = adfuller(series)
    l = ['diff' if ii else 'original', r[0], r[1]]
    for _, v in r[4].items():
        l.append(v)
    ll.append(l)
s = tabulate(ll, headers=[r'name', r'ADF stat', r'p-value', r'1%', r'5%', r'10%'], floatfmt=('10.5f', '10.5f', '10.5f', '10.5f', '10.5f', '10.5f'), tablefmt='latex_booktabs')
#open('tbl_adf_diff.txt', 'w').write(s)

ll = []
irr = dfi['irr'].to_numpy()
data = diff(log(irr)) 
for ii, series in enumerate((irr, data)):
    r = adfuller(series)
    l = ['diff-log' if ii else 'original', r[0], r[1]]
    for _, v in r[4].items():
        l.append(v)
    ll.append(l)
s = tabulate(ll, headers=[r'name', r'ADF stat', r'p-value', r'1%', r'5%', r'10%'], floatfmt=('10.5f', '10.5f', '10.5f', '10.5f', '10.5f', '10.5f'), tablefmt='latex_booktabs')
#open('tbl_adf_irr.txt', 'w').write(s)

#####################################################################
#
# 
#
#####################################################################

#dgs30 = dfd['dgs30'].to_numpy()
#data = diff(log(dgs30[-2800:]))
irr = dfi['irr'].to_numpy()
data = diff(log(irr)) 

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
ax[0].plot(irr, color='blue')
ax[0].set_ylabel('irr')
ax[1].plot(data)
ax[1].set_xlabel(r'No. of $\Delta t$')
ax[1].set_ylabel('diff(log(irr))')
fig.tight_layout()
fig.savefig('irr.pdf', bbox_inches='tight')

#####################################################################
#
#  Parameter Estimation: BS
#
#####################################################################

def dbs(x, par):
    mu, sigma = par
    return 1 / (sqrt(2 * pi) * sigma) * exp(-1 / 2 * ((x - mu) / sigma) ** 2)

def llh_bs(data, par):
    if par[1] > 0:
        return sum(log(dbs(data, par)))
    else:
        return -inf

par_bs = fmin(lambda x: -llh_bs(data, x), (0.1, 0.05), maxiter=5000)
print('fmin bs: ', -llh_bs(data, par_bs), par_bs)

def cs_fair1(x):
    return x[1] - 1e-5 
cs = ({'type': 'ineq', 'fun': cs_fair1},)
t = shgo(lambda x: -llh_bs(data, x), constraints=cs, bounds=([-1, 1], [1e-5, 1]), sampling_method='sobol')
if t['success']:
    par_bs = t['x']
    print('shgo bs: ', t['fun'], 'param bs: ', par_bs)

#####################################################################
#
#  Parameter Estimation: eNIG 
#
#####################################################################

def denig(x, par):
    # alpha >= 0, delta > 0, alpha ** 2 - beta ** 2 > 0
    alpha, beta, delta, mu = par
    return alpha * delta * exp(delta * sqrt(alpha ** 2 - beta ** 2) + beta * (x - mu)) * kv(1, alpha * sqrt(delta ** 2 + (x - mu) ** 2)) / (pi * sqrt(delta ** 2 + (x - mu) ** 2))

def llh_enig(data, par):
    if par[1] > 0:
        return sum(log(denig(data, par)))
    else:
        return -inf

par_enig = fmin(lambda x: -llh_enig(data, x), (1, 0.5, 0.5, 0), maxiter=5000)
print('fmin enig: ', -llh_enig(data, par_enig), par_enig)

def cs_fair(x):
    return x[0] ** 2 - x[1] ** 2 - 1e-5  
cs = ({'type': 'ineq', 'fun': cs_fair},)
t = shgo(lambda x: -llh_enig(data, x), constraints=cs, bounds=([0, 20], [-20, 20], [1e-10, 20], [-20, 20]), sampling_method='sobol')
if t['success']:
    par_enig = t['x']
    print('shgo enig: ', t['fun'], 'param enig: ', par_enig)

#####################################################################
#
#  Parameter Estimation: eVG
#
#####################################################################

def devg(x, par):
    c, sigma, theta, nu = par
    return 2.0 * exp(theta * (x - c) / sigma**2) / (sigma * sqrt(2.0 * pi) * nu**(1 / nu) * gamma(1 / nu)) * (abs(x - c) / sqrt(2 * sigma**2 / nu + theta**2))**(1 / nu - 0.5) * kv(1 / nu - 0.5, abs(x - c) * sqrt(2 * sigma**2 / nu + theta**2) / sigma**2)

def fit_moments(x):
    mu = mean(x)
    sigma_squared = mean((x - mu)**2)
    beta = mean((x - mu)**3) / mean((x - mu)**2)**1.5
    kapa = mean((x - mu)**4) / mean((x - mu)**2)**2
    sigma = sigma_squared**0.5
    nu = kapa / 3.0 - 1.0
    theta = sigma * beta / (3.0 * nu)
    c = mu - theta
    return (c, sigma, theta, nu)

def llh_evg(data, par):
    if (par[1] > 0) & (par[3] > 0):
        return sum(log(devg(data, par)))
    else:
        return -inf

def fit_evg(data):
    init = array(fit_moments(data))
    par = fmin(lambda x: -llh_evg(data, x), init, maxiter=5000)
    return list(par)

par_evg = fit_evg(data)
print('fmin evg: ', -llh_evg(data, par_evg), par_evg)

def cs_fair1(x):
    return x[1] - 1e-4 
def cs_fair3(x):
    return x[3] - 1e-4  
cs = ({'type': 'ineq', 'fun': cs_fair1}, {'type': 'ineq', 'fun': cs_fair3},)
t = shgo(lambda x: -llh_evg(data, x), constraints=cs, bounds=([-1, 1], [1e-4, 2], [-1, 1], [1e-4, 2]), sampling_method='sobol')
if t['success']:
    par_evg = t['x']
    print('shgo evg: ', t['fun'], 'param evg: ', par_evg)

#####################################################################
#
# 
#
#####################################################################

aic_bs = 2 * 2 - 2 * llh_bs(data, par_bs)

l = []
l.append(list(append(par_bs, aic_bs)))
s = tabulate(l, headers=[r'$\mu$', r'$\sigma$', r'AIC',], floatfmt=('15.10f', '15.10f', '15.10f',), tablefmt='latex_raw')
open('tbl_param_bs.txt', 'w').write(latex_booktabs(s).replace('rrr', 'ccc'))

aic_enig = 2 * 4 - 2 * llh_enig(data, par_enig)

l = []
l.append(list(append(par_enig, aic_enig)))
s = tabulate(l, headers=[r'$\alpha$', r'$\beta$', r'$\delta$', r'$\mu$', r'AIC',], floatfmt=('15.10f', '15.10f', '15.10f', '15.10f', '15.10f',), tablefmt='latex_raw')
open('tbl_param_enig.txt', 'w').write(latex_booktabs(s).replace('rrrrr', 'ccccc'))

aic_evg = 2 * 4 - 2 * llh_evg(data, par_evg)

l = []
l.append(list(append(par_evg, aic_evg)))
s = tabulate(l, headers=['$c$', r'$\sigma$', r'$\theta$', r'$\nu$', r'AIC',], floatfmt=('15.10f', '15.10f', '15.10f', '15.10f', '15.10f',), tablefmt='latex_raw')
open('tbl_param_evg.txt', 'w').write(latex_booktabs(s).replace('rrrrr', 'ccccc'))

#####################################################################
#
# 
#
#####################################################################

from scipy.stats import gaussian_kde
density = gaussian_kde(data)
xs = linspace(-0.3, 0.3, 200)
density.covariance_factor = lambda : .25
density._compute_covariance()

fig, ax = plt.subplots(1, 1, figsize=(9, 6))
plt.plot(xs, density(xs), 'k-', label='KDE')
plt.plot(xs, dbs(xs, par_bs), ':', color='green', label='BS')
plt.plot(xs, denig(xs, par_enig), '-.', color='orangered', label='eNIG') 
plt.plot(xs, devg(xs, par_evg), '--', color='blue', label='eVG')
ax.set_xlabel('Data')
ax.set_ylabel('Density')
ax.legend(loc='upper right', shadow=True)
fig.savefig('fit.pdf', bbox_inches='tight')

#####################################################################
#
# Reinv Computation
#
#####################################################################

import reinv

def create_tbl():
    for typ in ['bs', 'enig', 'evg']:
        setattr(reinv, 'cf', lambda dt: getattr(reinv, 'cf_%s' % typ)(dt, globals()['par_%s' % typ]))
        tbl = []
        for s0 in linspace(0.03, 0.05, 41):
            reinv.s0 = s0
            bdry, value = reinv.value_cos()
            r5, r6, r7, r8, r9 = bdry
            tbl.append([100 * s0, 100 * r5, 100 * r6, 100 * r7, 100 * r8, 100 * r9, 10000 * value])
        s = tabulate(tbl, headers=['$r_0$', r'$r_5^\star$', r'$r_6^\star$', r'$r_7^\star$', r'$r_8^\star$', r'$r_9^\star$', '$\psi_1$（bps）'], floatfmt=('10.2f', '10.6f', '10.6f', '10.6f', '10.6f', '10.2f', '10.6f'), tablefmt='latex_raw')
        open('tbl_reinv_%s.txt' % typ, 'w').write(latex_booktabs(s).replace('rrrrrrr', 'ccccccc'))
create_tbl()
