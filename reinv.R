library(ghyp)
library(xtable)

b <- TRUE;
if (b) {
  S <- read.csv(file='irr.txt', header=TRUE);
  S <- as.numeric(S$data);
  Dt <- S[1];
  S <- S[2:length(S)];
} else {
  S <- read.csv(file='DGS30.csv', header=FALSE);
  S <- as.numeric(S[7536:10505, 2]);
  S <- na.omit(0.01 * S);
  Dt <- 1 / 252;
}

Z <- na.omit(diff(log(S)));

enig.fit <- fit.NIGuv(Z);
pdf('fit_enig.pdf', width=10, height=5)
par(mfrow=c(1,3))
hist(enig.fit, ghyp.col='red', legend.cex=0.7, ylim=c(0,8), main='', xlab='Data')
hist(enig.fit, log.hist=T, nclass=30, plot.legend=F, ghyp.col='red', xlab='Data')
qqghyp(enig.fit, plot.legend=T, legend.cex=0.7, main='', ghyp.col='red')
dev.off()

evg.fit <- fit.VGuv(Z);
pdf('fit_evg.pdf', width=10, height=5)
par(mfrow=c(1,3))
hist(evg.fit, ghyp.col='blue', legend.cex=0.7, ylim=c(0,8), main='', xlab='Data')
hist(evg.fit, log.hist=T, nclass=30, plot.legend=F, ghyp.col='blue', xlab='Data')
qqghyp(evg.fit, plot.legend=T, legend.cex=0.7, main='', ghyp.col='blue')
dev.off()

#####################################################################
#
# 
#
#####################################################################

#library(yuima)
#library(ggplot2)

# GBM 
#gbm <- setYuima(model=setModel(drift='mu * x', diffusion='sigma * x', solve.variable='x'), data=setData(S, delta=Dt));
#gbm.fit <- qmle(gbm, start=list(mu=.5, sigma=.5), lower=list(mu=1e-15, sigma=1e-15), method='L-BFGS-B');
#gbm.coef <- coef(gbm.fit);

## GBM 
#gbm <- setYuima(model=setPoisson(df='dnorm(z, mu, sigma)', solve.variable='x'), data=setData(cumsum(Z), delta=Dt));
#gbm.fit <- qmle(gbm, start=list(mu=1, sigma=0.5), lower=list(mu=1e-20, sigma=1e-20), method='L-BFGS-B');
#gbm.coef <- coef(gbm.fit);

## eVG
#evg.fit <- qmle(evg, start=list(lambda=0.5, alpha=0.5, beta=0.1, mu=0), lower=list(lambda=1e-15, alpha=1e-10, beta=1e-10, mu=-1), method='L-BFGS-B');
#evg.coef <- coef(evg.fit);

## eNIG
#enig <- setYuima(model=setPoisson(df='dNIG(z, alpha, beta, delta, mu)', solve.variable='x'), data=setData(cumsum(Z), delta=Dt));
#enig.fit <- qmle(enig, start=list(alpha=0.5, beta=0.1, delta=0.1, mu=0), lower=list(alpha=1e-15, beta=1e-15, delta=1e-15, mu=-1), method='L-BFGS-B');
#enig.coef <- coef(enig.fit);

## Plotting
#fig <- ggplot(data.frame(x=Z), aes(x, colour='Data', linetype='Data')) + geom_density(na.rm=TRUE);
#fig <- fig + stat_function(fun=dnorm, args=list(mean=gbm.coef['mu'], sd=gbm.coef['sigma']), aes(colour='GBM', linetype='GBM'));
#fig <- fig + stat_function(fun=dNIG, args=list(alpha=enig.coef['alpha'], beta=enig.coef['beta'], delta=enig.coef['delta'], mu=enig.coef['mu']), aes(colour='eNIG', linetype='eNIG'));
#fig <- fig + stat_function(fun=dvgamma, args=list(lambda=evg.coef['lambda'], alpha=evg.coef['alpha'], beta=evg.coef['beta'], mu=evg.coef['mu']), aes(colour='eVG', linetype='eVG'));
#fig <- fig + scale_colour_manual('', values=c('black', 'green4', 'blue', 'red')); 
#fig <- fig + scale_linetype_manual('', values=c('solid', 'dotted', 'longdash', 'dotdash')); 
#fig <- fig + theme(legend.position='top');
#ggsave('fit.pdf', width=9, height=6);

#####################################################################
#
# 
#
#####################################################################

test_change <- function (X, N=10000, est=0) {
  #
  #  Nonparametric change point test for a univariate series using the 
  #  Kolmogorov-Smirnov statistic.
  #
  # Input
  #        X: (n x 1) vector of data (residuals or observations)
  #        N: number of bootstrap samples to compute the P-value
  #      est: 1 if tau is estimated, 0 otherwise (default).
  #
  #  Output
  #          KS: Kolmogorov-Smirnov statistic 
  #      pvalue: (#) calculated with N bootstrap samples
  #         tau: estimation of change point time
  #     tau_rel: estimation of change point relative time (percent of the
  #              sample size).
  #
  #   Ref: Remillard, B. (2012)  Non-Parametric Change Point Problems 
  #        Using Multipliers, SSRN Working Paper Series No. 2043632.
  
  n <- length(X)
  
  fct.ret <- changepoint(X, est)
  KS <- fct.ret$KS;
  tau <- fct.ret$tau;
  tau_rel <- fct.ret$tau_rel;
  
  KSsim <- mat.or.vec(N, 1);
  
  for (k in 1:N) {
    x <- runif(n);
    KSsim[k] <- changepoint(x, 0)$KS;
  }
  
  pvalue <- 100 * mean((KSsim > KS));
  return(list(KS=KS, pvalue=pvalue, tau=tau, tau_rel=tau_rel))
}

changepoint <- function(x, est) {
  n <- length(x);
  tau <- 0;
  tau_rel <- 0;
  
  R <- as.numeric(rank(x, ties.method = 'average'));
  P <- matrix(0, n, n);
  z <- P;
  
  for (k in 1:n) {
    z[,k] <- (R <= k) - k/n;
  }
  
  P <- apply(z, 2, cumsum); 
  P <- P / sqrt(n);  # process values!!
  
  M1 <- apply(abs(t(P)), 2, max);
  KS <- max(M1);
  
  if (est) {
    tau <- 1;
    while (M1[tau] < KS) {
      tau <- tau + 1;
    }
    tau_rel <- 100 * tau / n;
  }
  return(list(KS=KS, tau=tau, tau_rel=tau_rel))
}

df0 <- read.csv(file='df0.txt', header=TRUE);
df0 <- as.numeric(df0$data);
r <- test_change(df0);
rd <- test_change(na.omit(diff(df0)));
l <- list(name=c('original', 'diff'), KS=c(r$KS, rd$KS), pvalue=c(r$pvalue, rd$pvalue));
ll <- as.data.frame(l)
print(xtable(ll, align='lccc', 
    latex.environments='center',
    caption=''), 
    include.rownames=FALSE,
    file='/home/clarktu/usr/work/research/reinv/tbl_ks_diff.txt')

r <- test_change(S);
rd <- test_change(Z);
l <- list(name=c('original', 'diff-log'), KS=c(r$KS, rd$KS), pvalue=c(r$pvalue, rd$pvalue));
ll <- as.data.frame(l)
print(xtable(ll, align='lccc', 
             latex.environments='center',
             caption=''), 
      include.rownames=FALSE,
      file='/home/clarktu/usr/work/research/reinv/tbl_ks_irr.txt')
