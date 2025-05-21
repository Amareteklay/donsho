## Project plan

### Objective:
- To look at the interaction between shocks and emerging pests and pathogens (EPPs).

### Input:
- Data on shocks and EPPs

### Expected output:
- TBD

### Tasks:
- Data cleaning
- Exploratory analysis
- Regression analysis
- Visualization and mapping
- Reporting final results


### Outline
#### Aim
- Which crisis cause and are caused by EPP outbreaks, specifically outbreaks of human pathogens.


#### Data
- Shocks database: covers shocks from geographical, climate, war, agricultural production

- DON We are looking at events by year and country.


#### Analysis

- cooccurrence and lag between disease outbreaks and various categories of shocks to see if a certain subset of crisis types are more likely to cause disease outbreaks or vice versa.

#### Steps
- Compute temporal lag between disease outbreak and shocks (crisis)
- Negative lag implies disease occurs before crisis (Or is it the opposite?? Check!)
- Positive lag when disease occurs after a crisis (Or is it the opposite?? Check!)
- Which have more immediate (proximate causes) relationship and which have more distant causal effect?
- We are implicitly using the lag as indicating likelyhood of causation
- We should include cooccurrence...(shocks and diseases in the same year)
- Reproduce this in python:
            
        x11()
        m.emm.df %>%
        ggplot(aes(Shock.category, emmean, ymin=asymp.LCL, ymax=asymp.UCL)) +
        geom_pointrange() +
        ylab("Time from outbreak news (years)")+
        theme_bw()+ylim(-2,2)+coord_flip()+geom_hline(yintercept = 0)

- Percentage of times a crisis occurs with a disease outbreak
- The lag tells us the order of occurrence while the percentage tells us the frequency.
#### Analysis 1
- Lag
#### Analysis 2
- Cooccurrence
#### Analysis 3
- Regional variation in both analysis 1 and 2
### Analysis 4
- Use casesTotal and deaths from DONdatabase to measure severity
- Take into account population size