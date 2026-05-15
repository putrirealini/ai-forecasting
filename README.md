# KriyaLogic ML Forecasting Engine

## Overview

This module provides machine learning-based sales forecasting using Facebook Prophet. It includes data seeding from Excel files and forecast generation with comprehensive error handling.

## Features

### ✅ Error Handling & Robustness
- **MongoDB Connection Retry Logic**: Automatic retry with configurable attempts and delays
- **Connection Timeout**: 5-second timeout to prevent hanging
- **Try-Except Blocks**: Comprehensive error handling in all functions
- **Data Validation**: Validation before Prophet model fitting
- **Graceful Degradation**: Detailed error messages instead of silent failures

### ✅ Data Validation
- **Minimum Data Points**: Requires minimum 20 data points for Prophet (configurable)
- **Null Value Handling**: Automatic filling of null values using forward/backward fill
- **Negative Value Detection**: Warnings for unusual data patterns
- **Column Validation**: Ensures required columns exist before processing
- **File Validation**: Validates Excel file existence and readability

### ✅ Forecasting Features
- **30-Day Forecast**: Generates 30-day sales predictions by default
- **Confidence Intervals**: 95% confidence interval bounds (upper/lower)
- **Model Persistence**: Saves trained models using pickle
- **Batch Processing**: Can forecast multiple products in one run
- **Result Validation**: Ensures no NaN values in results

## Installation

### Prerequisites
- Python 3.8+
- MongoDB (local or cloud)
- `.env` file in `../kriyalogic-backend/` with `MONGO_URI`

### Setup

```bash
cd ai_engine
pip install -r requirements.txt
```

## Usage

### 1. Seed Data from Excel

```bash
python seed_data.py
```

**Requirements:**
- Excel file: `../data new sep-jan.xlsx`
- Columns: `Tanggal`, `Nama Patung`, `Total (Rp)`, `Nama Artisan`, `Tour Guide`, `Metode Pembayaran`

**Output:**
- Populates `saletransactions` collection
- Populates `saleitems` collection
- Detailed logging of success/failure

### 2. Generate Forecasts

#### Single Product
```bash
python ml_forecasting.py PB001
```

#### Multiple Products
```bash
python ml_forecasting.py PB001 PG001 PN001 PGW001 PAM001 PBA001 PH001
```

**Output:**
- Saves forecasts to `forecastresults` collection
- Saves model to `../models/{product_code}_model.pkl`
- Detailed logging of process

## Configuration

### Connection Settings
- `CONNECTION_TIMEOUT_MS`: 5000 (5 seconds)
- `RETRY_ATTEMPTS`: 3
- `RETRY_DELAY`: 2 seconds

### Forecasting Settings
- `MIN_DATA_POINTS`: 20 (minimum required for Prophet)
- `FORECAST_PERIOD`: 30 (days to forecast)

To modify, edit the constants in the scripts:

```python
MIN_DATA_POINTS = 20
FORECAST_PERIOD = 30
CONNECTION_TIMEOUT_MS = 5000
RETRY_ATTEMPTS = 3
```

## Output Format

### Forecast Results (MongoDB)

```json
{
  "_id": "ObjectId",
  "product_code": "PB001",
  "forecast_date": "2025-09-01",
  "predicted_quantity": 7423.45,
  "lower_bound_estimate": 6309.43,
  "upper_bound_estimate": 8537.47,
  "last_updated": "2026-05-02T10:30:45.123456"
}
```

## Logging & Debugging

All scripts provide detailed console output:

```
🚀 KriyaLogic ML Forecasting Engine
============================================================
🔌 Attempting MongoDB connection (attempt 1/3)...
✓ MongoDB connected successfully
🔮 Starting forecast for: PB001
📊 Fetching historical sales for product: PB001
✓ Found 150 historical data points
✓ Data validation passed (150 data points)
📈 Fitting Prophet model...
✓ Prophet model fitted successfully
✓ Model saved to ../models/PB001_model.pkl
🔮 Generating 30-day forecast...
✓ Forecast generated successfully
✓ Generated 30 valid forecast records
============================================================
```

### Error Examples

#### Connection Error
```
🔌 Attempting MongoDB connection (attempt 1/3)...
⚠ Connection failed: [Errno -2] Name or service not known
  Retrying in 2 seconds...
...
❌ Failed to connect after 3 attempts
```

#### Data Validation Error
```
📊 Fetching historical sales for product: PB001
⚠ No historical data found for PB001
❌ Data validation failed: No data available for PB001
```

#### Prophet Error
```
❌ Error fitting Prophet model: Insufficient samples for fitting seasonality
```

## Troubleshooting

### Issue: "MONGO_URI not found in .env file"
**Solution:** Create `.env` in `../kriyalogic-backend/` with:
```
MONGO_URI=mongodb://localhost:27017/kriyalogic
```

### Issue: "No historical data found"
**Solution:** 
1. Verify `saletransactions` and `saleitems` exist in MongoDB
2. Run `seed_data.py` first to populate with test data
3. Check that product codes match between sale data and forecast request

### Issue: "Insufficient data: X points (minimum 20 required)"
**Solution:**
1. Seed more data using `seed_data.py`
2. Reduce `MIN_DATA_POINTS` constant if needed (not recommended)
3. Product may have insufficient sales history

### Issue: Connection Timeout
**Solution:**
1. Verify MongoDB is running: `mongo` or connect via MongoDB Compass
2. Check MONGO_URI is correct
3. Increase `CONNECTION_TIMEOUT_MS` if using slow network
4. Check firewall/network connectivity

## Model Persistence

Trained Prophet models are automatically saved to:
```
../models/{product_code}_model.pkl
```

You can load and reuse models:

```python
import pickle

with open('../models/PB001_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Use model for new forecasts
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

## Performance Considerations

- **Data Loading**: ~1-2s for 150 data points
- **Prophet Fitting**: ~5-10s per product
- **Forecast Generation**: ~1s per product
- **Database Save**: ~0.5s per 30 forecasts

Total time for single product: ~10-15 seconds

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| pandas | >=1.3.0 | Data manipulation |
| prophet | >=1.1.0 | Time series forecasting |
| pymongo | >=4.0.0 | MongoDB connection |
| python-dotenv | >=0.19.0 | Environment variables |
| openpyxl | >=3.7.0 | Excel file reading |

## Future Enhancements

- [ ] Add Prophet hyperparameter tuning
- [ ] Implement forecast accuracy metrics
- [ ] Add support for external regressors (holidays, events)
- [ ] PostgreSQL support as alternative to MongoDB
- [ ] REST API wrapper for forecasting
- [ ] Automated daily forecast generation
- [ ] Web dashboard for forecast results

## License

MIT - See main repository README

## Support

For issues or questions, refer to the main Integration project README.
