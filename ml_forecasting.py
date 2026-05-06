import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
# from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutException
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta
import pickle
import traceback

# Load environment variables from parent directory
load_dotenv(dotenv_path='./.env')

# Configuration
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    print("❌ ERROR: MONGO_URI not found in .env file")
    sys.exit(1)

# Constants for data validation
MIN_DATA_POINTS = 20  # Minimum data points Prophet needs
FORECAST_PERIOD = 30  # Days to forecast
CONNECTION_TIMEOUT_MS = 5000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2  # seconds

# Global client variable
client = None
db = None

def connect_to_mongodb(retry_count=0):
    """
    Connect to MongoDB with retry logic and timeout.
    
    Args:
        retry_count: Current retry attempt number
        
    Returns:
        tuple: (client, db) or (None, None) if failed
    """
    global client, db
    
    try:
        print(f"🔌 Attempting MongoDB connection (attempt {retry_count + 1}/{RETRY_ATTEMPTS})...")
        
        client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=CONNECTION_TIMEOUT_MS,
            socketTimeoutMS=5000,
            connectTimeoutMS=CONNECTION_TIMEOUT_MS
        )
        
        # Test connection
        client.admin.command('ping')
        db = client.get_database()
        
        print("✓ MongoDB connected successfully")
        return client, db
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"⚠ Connection failed: {str(e)}")
        
        if retry_count < RETRY_ATTEMPTS - 1:
            print(f"  Retrying in {RETRY_DELAY} seconds...")
            import time
            time.sleep(RETRY_DELAY)
            return connect_to_mongodb(retry_count + 1)
        else:
            print(f"❌ Failed to connect after {RETRY_ATTEMPTS} attempts")
            return None, None
            
    except Exception as e:
        print(f"❌ Unexpected error during MongoDB connection: {e}")
        traceback.print_exc()
        return None, None

def fetch_historical_sales(parent_code):
    """
    Fetch historical sales data for a specific parent_code from MongoDB.
    Groups sales by date and sums the total sales for that product.
    
    Args:
        parent_code: Product code to fetch data for
        
    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'y' (sales) columns, or empty DataFrame
    """
    if client is None or db is None:
        print(f"❌ Database not connected")
        return pd.DataFrame()
    
    try:
        print(f"📊 Fetching historical sales for product: {parent_code}")
        
        pipeline = [
            {
                "$match": {"parentCodeSnapshot": parent_code}
            },
            {
                "$lookup": {
                    "from": "saletransactions",
                    "localField": "saleTransactionId",
                    "foreignField": "_id",
                    "as": "transaction"
                }
            },
            {
                "$unwind": "$transaction"
            },
            {
                "$group": {
                    "_id": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$transaction.soldAt"
                        }
                    },
                    "total_sales": {"$sum": "$sellingPriceSnapshot"}
                }
            },
            {
                "$sort": {"_id": 1}
            }
        ]

        result = list(db.saleitems.aggregate(pipeline))
        
        if not result or len(result) == 0:
            print(f"⚠ No historical data found for {parent_code}")
            return pd.DataFrame()
        
        print(f"✓ Found {len(result)} historical data points")

        # Convert to DataFrame
        data = [{"ds": item["_id"], "y": item["total_sales"]} for item in result]
        df = pd.DataFrame(data)
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df
        
    except Exception as e:
        print(f"❌ Error fetching historical sales: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def validate_forecast_data(df, parent_code):
    """
    Validate data before Prophet forecasting.
    
    Args:
        df: Input DataFrame
        parent_code: Product code for logging
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None or df.empty:
        return False, f"No data available for {parent_code}"
    
    # Check minimum data points
    if len(df) < MIN_DATA_POINTS:
        return False, f"Insufficient data: {len(df)} points (minimum {MIN_DATA_POINTS} required)"
    
    # Check for required columns
    if 'ds' not in df.columns or 'y' not in df.columns:
        return False, "Missing required columns: 'ds' and 'y'"
    
    # Check for null values
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"⚠ Warning: Found {null_count} null values, filling with forward fill")
        df['y'].fillna(method='ffill', inplace=True)
        df['y'].fillna(method='bfill', inplace=True)
    
    # Check for negative sales (unusual but might happen)
    negative_count = (df['y'] < 0).sum()
    if negative_count > 0:
        print(f"⚠ Warning: Found {negative_count} negative sales values")
    
    return True, None

def forecast_demand(parent_code):
    """
    Run Facebook Prophet to forecast demand for the next 30 days.
    
    Args:
        parent_code: Product code to forecast
        
    Returns:
        list: List of forecast dictionaries, or empty list if error
    """
    try:
        print(f"\n🔮 Starting forecast for: {parent_code}")
        
        # Fetch data
        df = fetch_historical_sales(parent_code)
        
        # Validate data
        is_valid, error_msg = validate_forecast_data(df, parent_code)
        if not is_valid:
            print(f"❌ Data validation failed: {error_msg}")
            return []
        
        print(f"✓ Data validation passed ({len(df)} data points)")
        
        # Fit Prophet model
        try:
            print("📈 Fitting Prophet model...")
            model = Prophet(
                daily_seasonality=False,
                yearly_seasonality=False,
                weekly_seasonality=True,
                interval_width=0.95  # 95% confidence interval
            )
            model.fit(df)
            print("✓ Prophet model fitted successfully")
            
        except Exception as e:
            print(f"❌ Error fitting Prophet model: {e}")
            traceback.print_exc()
            return []
        
        # Save model
        try:
            models_dir = '../models'
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'{parent_code}_model.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Model saved to {model_path}")
        except Exception as e:
            print(f"⚠ Warning: Could not save model: {e}")
        
        # Generate forecast
        try:
            print(f"🔮 Generating {FORECAST_PERIOD}-day forecast...")
            future = model.make_future_dataframe(periods=FORECAST_PERIOD)
            forecast = model.predict(future)
            print("✓ Forecast generated successfully")
        except Exception as e:
            print(f"❌ Error generating forecast: {e}")
            traceback.print_exc()
            return []
        
        # Format results
        forecasted = forecast.tail(FORECAST_PERIOD)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecasted['ds'] = forecasted['ds'].dt.strftime('%Y-%m-%d')
        
        # Validate forecast values
        last_updated = datetime.now().isoformat()
        results = []
        
        for _, row in forecasted.iterrows():
            # Check for NaN values
            if pd.isna(row['yhat']) or pd.isna(row['yhat_lower']) or pd.isna(row['yhat_upper']):
                print(f"⚠ Warning: NaN forecast value for {row['ds']}")
                continue
            
            # Ensure positive values (optional - adjust based on business logic)
            yhat = max(0, round(row['yhat'], 2))
            yhat_lower = max(0, round(row['yhat_lower'], 2))
            yhat_upper = max(0, round(row['yhat_upper'], 2))
            
            results.append({
                "product_code": parent_code,
                "forecast_date": row['ds'],
                "predicted_demand": yhat,
                "lower_bound_estimate": yhat_lower,
                "upper_bound_estimate": yhat_upper,
                "last_updated": last_updated
            })
        
        print(f"✓ Generated {len(results)} valid forecast records")
        return results
        
    except Exception as e:
        print(f"❌ Unexpected error in forecast_demand: {e}")
        traceback.print_exc()
        return []

def save_forecast_to_db(parent_code, forecast_data):
    """
    Save forecast results to MongoDB, overwriting existing data for the product.
    
    Args:
        parent_code: Product code
        forecast_data: List of forecast dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not db:
        print(f"❌ Database not connected")
        return False
    
    if not forecast_data or len(forecast_data) == 0:
        print(f"⚠ No forecast data to save for {parent_code}")
        return False
    
    try:
        collection = db.forecastresults
        
        # Delete existing forecasts for this product
        delete_result = collection.delete_many({"product_code": parent_code})
        print(f"✓ Deleted {delete_result.deleted_count} existing forecast records")
        
        # Insert new forecasts
        insert_result = collection.insert_many(forecast_data)
        print(f"✓ Inserted {len(insert_result.inserted_ids)} new forecast records for {parent_code}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving forecast to database: {e}")
        traceback.print_exc()
        return False

def main():
    """
    Main entry point for forecasting engine.
    """
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python ml_forecasting.py <parent_code> [parent_code2 ...]")
        print("Example: python ml_forecasting.py PB001 PG001 PN001")
        sys.exit(1)
    
    parent_codes = sys.argv[1:]
    
    print("=" * 60)
    print("🚀 KriyaLogic ML Forecasting Engine")
    print("=" * 60)
    
    # Connect to MongoDB
    global client, db
    client, db = connect_to_mongodb()
    
    if client is None or db is None:
        print("\n❌ Cannot proceed without database connection")
        sys.exit(1)
    
    try:
        # Process each product
        successful = 0
        failed = 0
        
        for parent_code in parent_codes:
            print(f"\n{'='*60}")
            
            # Run forecast
            forecast_data = forecast_demand(parent_code)
            
            # Save to database
            if forecast_data:
                if save_forecast_to_db(parent_code, forecast_data):
                    print(f"✓ Successfully completed forecast for: {parent_code}")
                    successful += 1
                else:
                    print(f"❌ Failed to save forecast for: {parent_code}")
                    failed += 1
            else:
                print(f"❌ Failed to generate forecast for: {parent_code}")
                failed += 1
        
        # Summary
        print(f"\n{'='*60}")
        print(f"📊 Forecasting Summary")
        print(f"{'='*60}")
        print(f"✓ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"Total: {len(parent_codes)}")
        print("=" * 60 + "\n")
        
        sys.exit(0 if failed == 0 else 1)
        
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error in main: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close database connection
        if client:
            client.close()
            print("✓ Database connection closed")

if __name__ == "__main__":
    main()
