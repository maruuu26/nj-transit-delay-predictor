# NJ Transit Delay Predictor

How predictable are NJ Transit delays before a train even runs? This project uses
historical stop-level data to estimate the chance that a train is 5+ minutes late at a
given station, line, hour, and day, and shows the patterns in an interactive dashboard.

## Data

- Source: the Kaggle dataset "NJ Transit + Amtrak (NEC) Rail Performance" (monthly CSVs of
  scraped stop-level train data, not official NJ Transit data).
- Used here: NJ Transit rows from March 2018 to December 2019. The 2020 months are left out
  because COVID service cuts changed how the system ran.
- Each row is one train at one stop. A stop counts as **late** when `delay_minutes >= 5`.
- To rerun it, download the dataset from Kaggle and unzip the monthly files into `data/archive/`.

## Approach

1. **Load and clean:** read the monthly files, keep NJ Transit rows that have a measured delay.
2. **Features:** only information known from the schedule: line, station, hour, day of week,
   month, weekend, and rush hour flags. Nothing from the actual run is used, so the model cannot
   peek at the outcome.
3. **Time-based split:** fit on older months, choose a decision threshold on a middle period,
   and test on the newest months (July to December 2019). A random split would let the model
   learn from the future.
4. **Models:** an "always on time" baseline, logistic regression, and gradient boosted trees.
5. **Metrics:** precision, recall, F1, ROC-AUC, and PR-AUC. Accuracy alone is misleading
   because most stops are on time.

Results are written to [`outputs/results.md`](outputs/results.md) when you run the training
script, along with charts in `outputs/`.

## Run it

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/train.py             # trains the models, saves results and charts to outputs/
streamlit run app.py            # opens the dashboard
```

`SAMPLE_FRAC` at the top of `src/train.py` controls how much of each monthly file is used
(0.25 by default so it runs in a few minutes; set it to 1.0 for all rows).

## Limitations

- The data is a community scrape of past trips, not a live feed or official records.
- The model sees only the schedule. It cannot know about weather, incidents, or what is
  happening on the track right now, so it estimates typical risk, not a specific train's fate.
- Cancelled trains are not modeled.
- Patterns from 2018 and 2019 may not match today's service.

## Project structure

```
src/train.py      load, clean, build features, train, evaluate, save outputs
app.py            Streamlit dashboard
outputs/          results, charts, summary tables, saved model
notebooks/        exploration notebooks
```

## AI assistance

I used Claude as a coding assistant on this project and am working back through each step in
`notebooks/` to learn the methods behind it.
