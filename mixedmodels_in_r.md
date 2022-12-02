Using MixedModels.jl in R
================
Jeroen van Paridon
2022-11-21

``` r
library(JuliaCall)
library(lme4)
```

    ## Loading required package: Matrix

``` r
library(tictoc)
```

## Why fit mixed-effects models with `MixedModels.jl` instead of `lme4`?

`lme4` has lots of great qualities: It’s free and easy to install, it’s
fairly straightforward to use if you know R, and its syntax and
functionality is stable (i.e. it’s no longer undergoing any major
changes). That last quality has downsides, too: Because `lme4` is no
longer being actively developed, any new work done on it consists of bug
fixes rather than major improvements. When all you want to do is fit a
reasonably *small, simple* linear or generalized linear mixed-effects
model, that lack of new development is a net positive. However, when
you’re trying to fit a harder model (e.g. using a very large dataset or
maximal model with many parameters) or the same model many times
(e.g. when doing some kind of bootstrapping) `lme4` can start to feel
very limited. Models may take very long to fit, or you may encounter
convergence issues.

`MixedModels.jl` solves a number of issues with `lme4`. Most of these
are not material to psychologists, but one that *does* matter is that
`MixedModels.jl` can fit models much faster than `lme4` because it is
written in `Julia`.[^1] This potential for speedup is the reason that
some of `lme4`’s contributors switched to working on `MixedModels.jl`,
despite `Julia` having a much smaller user-base than `R`.

Because `MixedModels.jl` is still undergoing active development, you can
expect new features to be added over time. Actively developed projects
sometimes break backward compatibility by e.g. removing features, but
`MixedModels.jl` uses semantic versioning, meaning that if large,
breaking changes are introduced, the developers will change the major
version number (e.g. if they are currently at v1.4.2, a breaking change
means they will go to v2.0.0) making it easy to see these changes coming
and revert to a compatible version, if necessary.

That said, `MixedModels.jl` is not an experimental package; several
companies use it commercially, so reliability is important and the
developers are thoughtful about making changes.

## Why you don’t need to switch to `Julia` to use `MixedModels.jl`

Despite the advantages that `Julia` offers, most psychologists are
probably not moving away from `R` anytime soon. There are simply too
many advantages to using `R`: Many people can read and write it, it has
a great package ecosystem for data cleaning, analysis, and visualization
and many people are familiar with these packages, and as a result many
existing data cleaning and modeling workflows are implemented in `R`. On
top of all that, learning a new language is a major time investment that
many people find hard to justify.

Luckily, there is no need to fully switch to `Julia` (or even learn much
`Julia` syntax) just to take advantage of `MixedModels.jl` model fitting
functionality. `R` has a package called `JuliaCall` that lets you run
`Julia` commands from the `R` command line, from RMarkdown documents,
and from inside RStudio. You can use `JuliaCall` to move data back and
forth between `Julia` and your `R` session, fit models using
`MixedModels.jl`, and do whatever else you may want to do in `Julia`
rather than `R`.

All that’s required to use `JuliaCall` is an existing `Julia` install on
your system, and a little bit of knowledge of whatever `Julia` command
you want to run.

## Installing Julia

Installing `Julia` is pretty easy. The package management system is less
error-prone than the `R` or `Python` system and you won’t really have to
think about it much while using `JuliaCall`.

To install `Julia` go to
[julialang.org/downloads](https://julialang.org/downloads/) and download
the **current stable release** for your system. If you have a MacBook
make sure to download the correct version for your processor
architecture (M1/M2 or Intel).

After installing, go to
[julialang.org/downloads/platform](https://julialang.org/downloads/platform/)
for some platform-specific instructions like how to add the `Julia`
binaries to your `PATH` variable. (This is important for using
`JuliaCall` later!)

## Setting up Julia environment

Now that you have `Julia` installed, all you need to do is install the
`JuliaCall` `R` package (e.g. through the RStudio interface) and then we
can set up the `Julia` environment.

``` r
# require JuliaCall
require(JuliaCall)

# set up and start the Julia instance
julia_setup()
```

    ## Julia version 1.8.3 at location /Applications/Julia-1.8.app/Contents/Resources/julia/bin will be used.

    ## Loading setup script for JuliaCall...

    ## Finish loading setup script for JuliaCall.

``` r
# install MixedModels.jl and JelyMe4.jl if necessary
julia_install_package_if_needed("MixedModels")
julia_install_package_if_needed("JellyMe4")
julia_install_package_if_needed("DataFrames")

# load Julia packages
julia_library("RCall")
```

``` r
julia_library("MixedModels")
```

``` r
julia_library("JellyMe4")
```

``` r
julia_library("DataFrames")
```

``` r
# load an example dataset from MixedModels.jl into R
julia_command("kb07 = DataFrame(MixedModels.dataset(:kb07));")
kb07 <- julia_eval("kb07")
```

## Fitting a big model in R

To set a baseline for the time cost of fitting a complex mixed-effects
model in `lme4`, I’m going to follow [an example from Doug
Bates](https://rpubs.com/dmbates/377897), one of the lead developers
behind both `lme4` and `MixedModels.jl`, and use the `kb07` dataset that
is included with `MixedModels.jl` for use in tutorials such as this one.
To get a nice, complex model for my demonstration, I’ll specify the
maximal model that’s licensed by the data generating process: All main
effects, all possible interactions between these main effects, and
random effects by both item and subject (including random intercepts and
random slopes for all main effect and interactions between main
effects).[^2]

I’ve already loaded the data from the `Julia` session into the `R`
session in the block of code above so that we can focus on model fitting
in the code below, but if you’re working in `R` and offloading model
fitting to `Julia`, you’ll generally want to work in the opposite
direction and load your data from the `R` session into the `JuliaCall`
session. How to do this will be demonstrated later in this vignette.

``` r
tic("lme4 model")
m <- lmer(rt_trunc ~ 1 + spkr * prec * load +
              (1 + spkr * prec * load | subj) +
              (1 + spkr * prec * load | item),
            kb07, REML=FALSE)
```

    ## boundary (singular) fit: see help('isSingular')

``` r
toc()
```

    ## lme4 model: 112.515 sec elapsed

``` r
summary(m)
```

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: rt_trunc ~ 1 + spkr * prec * load + (1 + spkr * prec * load |  
    ##     subj) + (1 + spkr * prec * load | item)
    ##    Data: kb07
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##  28740.3  29185.0 -14289.2  28578.3     1708 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.0488 -0.5755 -0.1435  0.3950  4.7448 
    ## 
    ## Random effects:
    ##  Groups   Name                         Variance Std.Dev. Corr             
    ##  subj     (Intercept)                  101208   318.1                     
    ##           spkrold                      165927   407.3    -0.06            
    ##           precmaintain                  67947   260.7    -0.63  0.55      
    ##           loadyes                      105316   324.5    -0.23  0.25  0.80
    ##           spkrold:precmaintain         174477   417.7     0.04 -0.99 -0.52
    ##           spkrold:loadyes              157590   397.0    -0.19 -0.66 -0.21
    ##           precmaintain:loadyes         115144   339.3     0.22 -0.19 -0.78
    ##           spkrold:precmaintain:loadyes 248708   498.7     0.11  0.54  0.31
    ##  item     (Intercept)                  285831   534.6                     
    ##           spkrold                       35599   188.7     0.13            
    ##           precmaintain                 208832   457.0    -0.93  0.00      
    ##           loadyes                      100905   317.7    -0.03  0.69 -0.06
    ##           spkrold:precmaintain          49655   222.8    -0.12 -0.85  0.17
    ##           spkrold:loadyes              165801   407.2    -0.02 -0.75  0.17
    ##           precmaintain:loadyes         138709   372.4     0.09 -0.37  0.07
    ##           spkrold:precmaintain:loadyes 320389   566.0    -0.05  0.61 -0.16
    ##  Residual                              401628   633.7                     
    ##                         
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.20                  
    ##  -0.24  0.59            
    ##  -0.99  0.14  0.15      
    ##   0.43 -0.44 -0.96 -0.34
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.89                  
    ##  -0.83  0.92            
    ##  -0.93  0.68  0.67      
    ##   0.80 -0.90 -0.96 -0.68
    ##                         
    ## Number of obs: 1789, groups:  subj, 56; item, 32
    ## 
    ## Fixed effects:
    ##                              Estimate Std. Error t value
    ## (Intercept)                   2347.19     111.99  20.959
    ## spkrold                        189.10      87.58   2.159
    ## precmaintain                  -586.54     106.46  -5.509
    ## loadyes                        158.15      92.89   1.702
    ## spkrold:precmaintain          -180.78     108.93  -1.660
    ## spkrold:loadyes                -20.12     123.19  -0.163
    ## precmaintain:loadyes           -75.52     116.50  -0.648
    ## spkrold:precmaintain:loadyes   187.35     169.78   1.103
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) spkrld prcmnt loadys spkrld:p spkrld:l prcmn:
    ## spkrold     -0.156                                              
    ## precmaintan -0.822  0.304                                       
    ## loadyes     -0.228  0.454  0.279                                
    ## spkrld:prcm  0.121 -0.810 -0.351 -0.420                         
    ## spkrld:ldys  0.090 -0.677 -0.093 -0.657  0.592                  
    ## prcmntn:ldy  0.212 -0.302 -0.358 -0.829  0.451    0.495         
    ## spkrld:prc: -0.103  0.510  0.169  0.593 -0.670   -0.838   -0.641
    ## optimizer (nloptwrap) convergence code: 0 (OK)
    ## boundary (singular) fit: see help('isSingular')

This model takes quite a bit of time to fit in `R`, around 120 seconds.
Two minutes is generally not prohibitive when fitting a single model,
but imagine this model had even more parameters or a much larger
dataset, or even worse, you wanted to use bootstrapping to get
confidence intervals for your parameter estimates and had to refit this
model 1000 times. Instead of two minutes, you would be looking at closer
to 2000 minutes of model fitting, more than 30 hours to get CIs for a
single model![^3]

## Fitting the model in Julia

``` r
# the dataframe is already present in the Julia instance
# because that's where we retrieved it from
# but in case you want to use data from an R dataframe, this is how to get it
julia_assign("kb07", kb07)

# specify and fit the model in Julia using MixedModels.jl
tic("MixedModels.jl model")
julia_command("jm = fit!(LinearMixedModel(@formula(
  rt_trunc ~ 1 + spkr * prec * load +
    (1 + spkr * prec * load | subj) +
    (1 + spkr * prec * load | item)), kb07), REML=false, progress=false);")
toc()
```

    ## MixedModels.jl model: 14.082 sec elapsed

``` r
# retrieve the model from Julia
jm <- julia_eval("robject(:lmerMod, Tuple([jm, kb07]));", need_return = "R")
summary(jm)
```

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: 
    ## rt_trunc ~ 1 + spkr + prec + load + spkr:prec + spkr:load + prec:load +  
    ##     spkr:prec:load + (1 + spkr + prec + spkr:prec + load + spkr:load +  
    ##     prec:load + spkr:prec:load | subj) + (1 + spkr + prec + spkr:prec +  
    ##     load + spkr:load + prec:load + spkr:prec:load | item)
    ##    Data: jellyme4_data
    ## Control: 
    ## lme4::lmerControl(optCtrl = list(maxeval = 1), optimizer = "nloptwrap",  
    ##     calc.derivs = FALSE, check.nobs.vs.rankZ = "warning", check.nobs.vs.nlev = "warning",  
    ##     check.nlev.gtreq.5 = "ignore", check.nlev.gtr.1 = "warning",  
    ##     check.nobs.vs.nRE = "warning", check.rankX = "message+drop.cols",  
    ##     check.scaleX = "warning", check.formula.LHS = "stop")
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##  28923.8  29368.4 -14380.9  28761.8     1708 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3219 -0.5865 -0.1359  0.3919  4.7284 
    ## 
    ## Random effects:
    ##  Groups   Name                         Variance Std.Dev. Corr             
    ##  subj     (Intercept)                  104196   322.8                     
    ##           spkrold                      170732   413.2    -0.06            
    ##           precmaintain                  69990   264.6    -0.63  0.55      
    ##           loadyes                      179511   423.7     0.04 -0.99 -0.52
    ##           spkrold:precmaintain         108387   329.2    -0.23  0.26  0.80
    ##           spkrold:loadyes              162457   403.1    -0.19 -0.67 -0.21
    ##           precmaintain:loadyes         118490   344.2     0.23 -0.19 -0.78
    ##           spkrold:precmaintain:loadyes 255791   505.8     0.11  0.54  0.31
    ##  item     (Intercept)                  293436   541.7                     
    ##           spkrold                       36475   191.0     0.13            
    ##           precmaintain                 214304   462.9    -0.93  0.00      
    ##           loadyes                       50669   225.1    -0.12 -0.84  0.17
    ##           spkrold:precmaintain         103226   321.3    -0.03  0.68 -0.06
    ##           spkrold:loadyes              169332   411.5    -0.02 -0.75  0.17
    ##           precmaintain:loadyes         142205   377.1     0.09 -0.37  0.07
    ##           spkrold:precmaintain:loadyes 327383   572.2    -0.05  0.61 -0.16
    ##  Residual                              412844   642.5                     
    ##                         
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.20                  
    ##   0.59 -0.25            
    ##   0.14 -0.99  0.15      
    ##  -0.44  0.43 -0.96 -0.34
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.89                  
    ##   0.92 -0.83            
    ##   0.68 -0.92  0.67      
    ##  -0.90  0.80 -0.96 -0.68
    ##                         
    ## Number of obs: 1789, groups:  subj, 56; item, 32
    ## 
    ## Fixed effects:
    ##                              Estimate Std. Error t value
    ## (Intercept)                   2346.55     113.50  20.674
    ## spkrold                        189.75      88.79   2.137
    ## precmaintain                  -585.89     107.90  -5.430
    ## loadyes                        158.79      92.11   1.724
    ## spkrold:precmaintain          -182.79     112.09  -1.631
    ## spkrold:loadyes                -20.76     124.80  -0.166
    ## precmaintain:loadyes           -76.17     118.07  -0.645
    ## spkrold:precmaintain:loadyes   189.36     171.97   1.101
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) spkrld prcmnt loadys spkrld:p spkrld:l prcmn:
    ## spkrold     -0.155                                              
    ## precmaintan -0.822  0.304                                       
    ## loadyes     -0.210 -0.292  0.138                                
    ## spkrld:prcm  0.099 -0.177 -0.224 -0.423                         
    ## spkrld:ldys  0.089 -0.677 -0.092  0.067 -0.023                  
    ## prcmntn:ldy  0.213 -0.302 -0.358 -0.140 -0.137    0.494         
    ## spkrld:prc: -0.103  0.509  0.168 -0.102 -0.078   -0.838   -0.641
    ## optimizer (LN_BOBYQA) convergence code: 5 (fit with MixedModels.jl)
    ## boundary (singular) fit: see help('isSingular')

## Wrap it in a function

To make calling `MixedModels.jl` from `R` more convenient (and not have
to remember a lot of `Julia` syntax!) we can wrap this process in a
function call. (I’m borrowing this idea and most of the following code
from [Phillip Alday](https://github.com/palday), one of the
`MixedModels.jl` developers)

``` r
jmer <- function(formula, data, REML = TRUE, progress = TRUE) {
    jf <- deparse1(formula)
    jreml = ifelse(REML, "true", "false")
    jprog = ifelse(progress, "true", "false")
    julia_assign("jmerdat", data)
    julia_command(sprintf(
      "jmermod = fit!(LinearMixedModel(@formula(%s), jmerdat), REML=%s, progress=%s);",
      jf, jreml, jprog
    ))
    julia_eval("robject(:lmerMod, Tuple([jmermod, jmerdat]));", need_return = "R")
}

tic("jmer model")
jm <- jmer(rt_trunc ~ 1 + spkr * prec * load +
             (1 + spkr * prec * load | subj) +
             (1 + spkr * prec * load | item),
            kb07, REML = FALSE, progress = FALSE)
toc()
```

    ## jmer model: 6.117 sec elapsed

``` r
summary(jm)
```

    ## Linear mixed model fit by maximum likelihood  ['lmerMod']
    ## Formula: 
    ## rt_trunc ~ 1 + spkr + prec + load + spkr:prec + spkr:load + prec:load +  
    ##     spkr:prec:load + (1 + spkr + prec + spkr:prec + load + spkr:load +  
    ##     prec:load + spkr:prec:load | subj) + (1 + spkr + prec + spkr:prec +  
    ##     load + spkr:load + prec:load + spkr:prec:load | item)
    ##    Data: jellyme4_data
    ## Control: 
    ## lme4::lmerControl(optCtrl = list(maxeval = 1), optimizer = "nloptwrap",  
    ##     calc.derivs = FALSE, check.nobs.vs.rankZ = "warning", check.nobs.vs.nlev = "warning",  
    ##     check.nlev.gtreq.5 = "ignore", check.nlev.gtr.1 = "warning",  
    ##     check.nobs.vs.nRE = "warning", check.rankX = "message+drop.cols",  
    ##     check.scaleX = "warning", check.formula.LHS = "stop")
    ## 
    ##      AIC      BIC   logLik deviance df.resid 
    ##  28923.8  29368.4 -14380.9  28761.8     1708 
    ## 
    ## Scaled residuals: 
    ##     Min      1Q  Median      3Q     Max 
    ## -3.3219 -0.5865 -0.1359  0.3919  4.7284 
    ## 
    ## Random effects:
    ##  Groups   Name                         Variance Std.Dev. Corr             
    ##  subj     (Intercept)                  104196   322.8                     
    ##           spkrold                      170732   413.2    -0.06            
    ##           precmaintain                  69990   264.6    -0.63  0.55      
    ##           loadyes                      179511   423.7     0.04 -0.99 -0.52
    ##           spkrold:precmaintain         108387   329.2    -0.23  0.26  0.80
    ##           spkrold:loadyes              162457   403.1    -0.19 -0.67 -0.21
    ##           precmaintain:loadyes         118490   344.2     0.23 -0.19 -0.78
    ##           spkrold:precmaintain:loadyes 255791   505.8     0.11  0.54  0.31
    ##  item     (Intercept)                  293436   541.7                     
    ##           spkrold                       36475   191.0     0.13            
    ##           precmaintain                 214304   462.9    -0.93  0.00      
    ##           loadyes                       50669   225.1    -0.12 -0.84  0.17
    ##           spkrold:precmaintain         103226   321.3    -0.03  0.68 -0.06
    ##           spkrold:loadyes              169332   411.5    -0.02 -0.75  0.17
    ##           precmaintain:loadyes         142205   377.1     0.09 -0.37  0.07
    ##           spkrold:precmaintain:loadyes 327383   572.2    -0.05  0.61 -0.16
    ##  Residual                              412844   642.5                     
    ##                         
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.20                  
    ##   0.59 -0.25            
    ##   0.14 -0.99  0.15      
    ##  -0.44  0.43 -0.96 -0.34
    ##                         
    ##                         
    ##                         
    ##                         
    ##  -0.89                  
    ##   0.92 -0.83            
    ##   0.68 -0.92  0.67      
    ##  -0.90  0.80 -0.96 -0.68
    ##                         
    ## Number of obs: 1789, groups:  subj, 56; item, 32
    ## 
    ## Fixed effects:
    ##                              Estimate Std. Error t value
    ## (Intercept)                   2346.55     113.50  20.674
    ## spkrold                        189.75      88.79   2.137
    ## precmaintain                  -585.89     107.90  -5.430
    ## loadyes                        158.79      92.11   1.724
    ## spkrold:precmaintain          -182.79     112.09  -1.631
    ## spkrold:loadyes                -20.76     124.80  -0.166
    ## precmaintain:loadyes           -76.17     118.07  -0.645
    ## spkrold:precmaintain:loadyes   189.36     171.97   1.101
    ## 
    ## Correlation of Fixed Effects:
    ##             (Intr) spkrld prcmnt loadys spkrld:p spkrld:l prcmn:
    ## spkrold     -0.155                                              
    ## precmaintan -0.822  0.304                                       
    ## loadyes     -0.210 -0.292  0.138                                
    ## spkrld:prcm  0.099 -0.177 -0.224 -0.423                         
    ## spkrld:ldys  0.089 -0.677 -0.092  0.067 -0.023                  
    ## prcmntn:ldy  0.213 -0.302 -0.358 -0.140 -0.137    0.494         
    ## spkrld:prc: -0.103  0.509  0.168 -0.102 -0.078   -0.838   -0.641
    ## optimizer (LN_BOBYQA) convergence code: 5 (fit with MixedModels.jl)
    ## boundary (singular) fit: see help('isSingular')

This model fit took even less time than the previous `MixedModels.jl`
call, because the `Julia` JIT compiler had already compiled this model
during this session and is smart enough to not recompile it when we call
it again. The time consumed here is therefore almost purely model
fitting time, and that turns out to be just \~7 seconds in
`MixedModels.jl`. That reduces the time cost of 1000 refits for
bootstrapping to 7000 seconds, or just 2 hours, rather than 30 hours
when using `lme4`.

[^1]: The main reason that a `Julia` package can be much faster than an
    `R` package is that `R` is *interpreted* whereas `Julia` is
    *compiled*. Broadly speaking, an interpreted language takes the code
    you write and does a fast but suboptimal translation to low-level
    machine instructions every time you run the code, while a compiled
    language takes a block of code and carefully optimizes it to very
    fast machine code. This optimization takes some time, which means
    that running code just once is often slower in a compiled language.
    The optimized code is stored however, so that when you run a block
    of code multiple times the speedup can be significant. Since fitting
    a mixed-effects model generally involves running a block of model
    fitting code hundreds or thousands of times, switching to a compiled
    language like `Julia` comes with a big speed advantage.

[^2]: To quote Doug: “Let me be clear that I do not advocate fitting
    such models”, I’m just fitting the maximal model as a stand-in for
    any other complex models that you may want to fit.

[^3]: Obviously the scenario I’m presenting here may not apply to you.
    You may only fit very minimal models, in which case `lme4` probably
    works just fine for you. Alternatively, 30 hours may not sound like
    a very long time to you, because you regularly fit much more
    complicated models and do a lot of bootstrap resampling! In that
    case, `MixedModels.jl` most definitely *is* for you. Always make a
    case-by-case evaluation when deciding which modeling tools to use.
    Is the additional hassle worth it for the speed advantage, yes or
    no?
