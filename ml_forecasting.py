import os
import sys
import argparse
from dotenv import load_dotenv
from pymongo import MongoClient
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
MIN_DATA_POINTS = 2  # Minimum data points Prophet needs
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

def build_forecast_dataframe(result):
    data = [{"ds": item["_id"], "y": item["total_quantity"]} for item in result]
    df = pd.DataFrame(data)
    df['ds'] = pd.to_datetime(df['ds'])
    return df

def get_month_bounds(year, month):
    start = datetime(year, month, 1)
    if month == 12:
        end = datetime(year + 1, 1, 1)
    else:
        end = datetime(year, month + 1, 1)
    return start, end

def fetch_pos_order_sales(parent_code, history_end_date=None):
    date_match = {"paid_at_parsed": {"$ne": None}}
    if history_end_date is not None:
        date_match["paid_at_parsed"]["$lt"] = history_end_date

    pipeline = [
        {
            "$match": {"status": "paid"}
        },
        {
            "$unwind": "$items"
        },
        {
            "$lookup": {
                "from": "masterproducts",
                "localField": "items.masterProductId",
                "foreignField": "_id",
                "as": "master_product"
            }
        },
        {
            "$unwind": "$master_product"
        },
        {
            "$match": {"master_product.parentCode": parent_code}
        },
        {
            "$addFields": {
                "paid_at_parsed": {
                    "$convert": {
                        "input": "$paidAt",
                        "to": "date",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        },
        {
            "$match": date_match
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$paid_at_parsed"
                    }
                },
                "total_quantity": {"$sum": {"$ifNull": ["$items.qty", 1]}}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]

    return list(db.posorders.aggregate(pipeline))

def fetch_legacy_sale_item_sales(parent_code, history_end_date=None):
    date_match = {"sold_at_parsed": {"$ne": None}}
    if history_end_date is not None:
        date_match["sold_at_parsed"]["$lt"] = history_end_date

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
            "$addFields": {
                "sold_at_parsed": {
                    "$convert": {
                        "input": "$transaction.soldAt",
                        "to": "date",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        },
        {
            "$match": date_match
        },
        {
            "$group": {
                "_id": {
                    "$dateToString": {
                        "format": "%Y-%m-%d",
                        "date": "$sold_at_parsed"
                    }
                },
                "total_quantity": {"$sum": {"$ifNull": ["$quantity", 1]}}
            }
        },
        {
            "$sort": {"_id": 1}
        }
    ]

    return list(db.saleitems.aggregate(pipeline))

def fetch_parent_codes(history_end_date=None):
    date_match = {"paid_at_parsed": {"$ne": None}}
    if history_end_date is not None:
        date_match["paid_at_parsed"]["$lt"] = history_end_date

    pipeline = [
        {
            "$match": {"status": "paid"}
        },
        {
            "$unwind": "$items"
        },
        {
            "$lookup": {
                "from": "masterproducts",
                "localField": "items.masterProductId",
                "foreignField": "_id",
                "as": "master_product"
            }
        },
        {
            "$unwind": "$master_product"
        },
        {
            "$addFields": {
                "paid_at_parsed": {
                    "$convert": {
                        "input": "$paidAt",
                        "to": "date",
                        "onError": None,
                        "onNull": None
                    }
                }
            }
        },
        {
            "$match": date_match
        },
        {
            "$group": {"_id": "$master_product.parentCode"}
        },
        {
            "$sort": {"_id": 1}
        }
    ]

    return [item["_id"] for item in db.posorders.aggregate(pipeline) if item.get("_id")]

def fetch_historical_sales(parent_code, history_end_date=None):
    """
    Fetch historical sales data for a specific parent_code from MongoDB.
    Groups sales by date and sums the total quantity for that product.
    
    Args:
        parent_code: Product code to fetch data for
        
    Returns:
        pd.DataFrame: DataFrame with 'ds' (date) and 'y' (quantity) columns, or empty DataFrame
    """
    if db is None:
        print(f"❌ Database not connected")
        return pd.DataFrame()
    
    try:
        print(f"📊 Fetching POS historical sales for product: {parent_code}")

        result = fetch_pos_order_sales(parent_code, history_end_date)

        if result and len(result) > 0:
            print(f"✓ Found {len(result)} POS historical data points")
            return build_forecast_dataframe(result)

        print(f"⚠ No POS historical data found for {parent_code}; checking legacy saleitems")
        result = fetch_legacy_sale_item_sales(parent_code, history_end_date)
        
        if not result or len(result) == 0:
            print(f"⚠ No historical data found for {parent_code}")
            return pd.DataFrame()
        
        print(f"✓ Found {len(result)} legacy historical data points")
        return build_forecast_dataframe(result)
        
    except Exception as e:
        print(f"❌ Error fetching historical sales: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def build_constant_forecast(parent_code, df, forecast_dates):
    quantity = max(0, int(round(df['y'].mean())))
    last_updated = datetime.now().isoformat()

    return [
        {
            "product_code": parent_code,
            "forecast_date": forecast_date.to_pydatetime(),
            "predicted_quantity": quantity,
            "lower_bound_estimate": quantity,
            "upper_bound_estimate": quantity,
            "last_updated": last_updated
        }
        for forecast_date in forecast_dates
    ]

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

def forecast_quantity(parent_code, target_start=None, target_end=None):
    """
    Run Facebook Prophet to forecast quantity for the next 30 days.
    
    Args:
        parent_code: Product code to forecast
        
    Returns:
        list: List of forecast dictionaries, or empty list if error
    """
    try:
        print(f"\n🔮 Starting forecast for: {parent_code}")
        
        # Fetch data
        target_mode = target_start is not None and target_end is not None
        df = fetch_historical_sales(parent_code, target_start if target_mode else None)

        if target_mode and df is not None and not df.empty and len(df) < MIN_DATA_POINTS:
            forecast_dates = pd.date_range(
                start=target_start,
                end=target_end - timedelta(days=1),
                freq='D'
            )
            print(
                f"⚠ Only {len(df)} historical point found for {parent_code}; "
                "using flat average forecast for target month"
            )
            results = build_constant_forecast(parent_code, df, forecast_dates)
            print(f"✓ Generated {len(results)} flat forecast records")
            return results
        
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
            if target_mode:
                days = (target_end - target_start).days
                print(
                    f"🔮 Generating target forecast for "
                    f"{target_start.strftime('%B %Y')} ({days} days)..."
                )
                future = pd.DataFrame({
                    'ds': pd.date_range(
                        start=target_start,
                        end=target_end - timedelta(days=1),
                        freq='D'
                    )
                })
            else:
                print(f"🔮 Generating {FORECAST_PERIOD}-day forecast...")
                future = model.make_future_dataframe(periods=FORECAST_PERIOD)
            forecast = model.predict(future)
            print("✓ Forecast generated successfully")
        except Exception as e:
            print(f"❌ Error generating forecast: {e}")
            traceback.print_exc()
            return []
        
        # Format results
        forecasted = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        if not target_mode:
            forecasted = forecasted.tail(FORECAST_PERIOD)
        
        # Validate forecast values
        last_updated = datetime.now().isoformat()
        results = []
        
        for _, row in forecasted.iterrows():
            # Check for NaN values
            if pd.isna(row['yhat']) or pd.isna(row['yhat_lower']) or pd.isna(row['yhat_upper']):
                print(f"⚠ Warning: NaN forecast value for {row['ds'].date()}")
                continue
            
            # Stock quantities should be whole units, not decimal estimates.
            yhat = max(0, int(round(row['yhat'])))
            yhat_lower = max(0, int(round(row['yhat_lower'])))
            yhat_upper = max(0, int(round(row['yhat_upper'])))
            
            results.append({
                "product_code": parent_code,
                "forecast_date": row['ds'].to_pydatetime(),
                "predicted_quantity": yhat,
                "lower_bound_estimate": yhat_lower,
                "upper_bound_estimate": yhat_upper,
                "last_updated": last_updated
            })
        
        print(f"✓ Generated {len(results)} valid forecast records")
        return results
        
    except Exception as e:
        print(f"❌ Unexpected error in forecast_quantity: {e}")
        traceback.print_exc()
        return []

def clear_forecast_period(start_date, end_date):
    if db is None:
        print(f"❌ Database not connected")
        return False

    try:
        delete_result = db.forecastresults.delete_many({
            "forecast_date": {
                "$gte": start_date,
                "$lt": end_date
            }
        })
        print(
            f"✓ Cleared {delete_result.deleted_count} existing forecast records "
            f"for {start_date.strftime('%B %Y')}"
        )
        return True
    except Exception as e:
        print(f"❌ Error clearing forecast period: {e}")
        traceback.print_exc()
        return False

def save_forecast_to_db(parent_code, forecast_data, replace_existing=True):
    """
    Save forecast results to MongoDB, overwriting existing data for the product.
    
    Args:
        parent_code: Product code
        forecast_data: List of forecast dictionaries
        
    Returns:
        bool: True if successful, False otherwise
    """
    if db is None:
        print(f"❌ Database not connected")
        return False
    
    if not forecast_data or len(forecast_data) == 0:
        print(f"⚠ No forecast data to save for {parent_code}")
        return False
    
    try:
        collection = db.forecastresults
        
        if replace_existing:
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

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate KriyaLogic product demand forecasts."
    )
    parser.add_argument(
        "parent_codes",
        nargs="*",
        help="Product codes to forecast. Omit when using --month and --year to forecast all POS products."
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        metavar="1-12",
        help="Target forecast month."
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Target forecast year."
    )

    args = parser.parse_args()

    if (args.month is None) != (args.year is None):
        parser.error("--month and --year must be provided together")

    if args.month is None and len(args.parent_codes) == 0:
        parser.error("provide parent codes or use --month and --year")

    if args.year is not None and args.year < 1900:
        parser.error("--year must be a valid year")

    return args

def main():
    """
    Main entry point for forecasting engine.
    """
    args = parse_args()
    
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
        target_start = None
        target_end = None
        replace_existing = True

        if args.month is not None and args.year is not None:
            target_start, target_end = get_month_bounds(args.year, args.month)
            print(
                f"🎯 Target forecast month: "
                f"{target_start.strftime('%B %Y')}"
            )
            print(
                f"📚 Historical cutoff: before "
                f"{target_start.strftime('%Y-%m-%d')}"
            )

            parent_codes = args.parent_codes or fetch_parent_codes(target_start)
            if not parent_codes:
                print("❌ No POS products found for the target historical period")
                sys.exit(1)

            if not clear_forecast_period(target_start, target_end):
                sys.exit(1)
            replace_existing = False
        else:
            parent_codes = args.parent_codes

        # Process each product
        successful = 0
        failed = 0
        
        for parent_code in parent_codes:
            print(f"\n{'='*60}")
            
            # Run forecast
            forecast_data = forecast_quantity(parent_code, target_start, target_end)
            
            # Save to database
            if forecast_data:
                if save_forecast_to_db(parent_code, forecast_data, replace_existing):
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
