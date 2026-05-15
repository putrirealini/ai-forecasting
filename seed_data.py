import os
import sys
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import pandas as pd
from datetime import datetime
import traceback

# Load environment variables
load_dotenv(dotenv_path='../kriyalogic-backend/.env')

# Configuration
MONGO_URI = os.getenv('MONGO_URI')
if not MONGO_URI:
    print("❌ ERROR: MONGO_URI not found in .env file")
    sys.exit(1)

CONNECTION_TIMEOUT_MS = 5000
RETRY_ATTEMPTS = 3
RETRY_DELAY = 2

# Global client and db
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

def validate_excel_file(file_path):
    """
    Validate that Excel file exists and can be read.
    
    Args:
        file_path: Path to Excel file
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Attempt to read file
        test_df = pd.read_excel(file_path, nrows=1)
        if test_df.empty:
            print(f"⚠ Warning: Excel file appears empty: {file_path}")
        return True
    except Exception as e:
        print(f"❌ Error reading Excel file: {e}")
        return False

def seed_data_from_excel():
    """
    Seed data from Excel to MongoDB for testing forecasting.
    With error handling and data validation.
    """
    try:
        # Hardcoded file path - adjust as needed
        excel_file = '../data/data new sep-jan.xlsx'
        
        print(f"📊 Seed Data Pipeline Started")
        print("=" * 60)
        
        # Validate file exists
        if not validate_excel_file(excel_file):
            print(f"❌ Cannot proceed without valid Excel file")
            return False
        
        # Load Excel data
        try:
            print(f"📂 Loading Excel file: {excel_file}")
            all_sheets = pd.read_excel(excel_file, sheet_name=None)
            df = pd.concat(all_sheets.values(), ignore_index=True)
            print(f"✓ Loaded {len(df)} rows from {len(all_sheets)} sheets")
        except Exception as e:
            print(f"❌ Error loading Excel file: {e}")
            traceback.print_exc()
            return False

        # Validate required columns
        required_cols = ['Tanggal', 'Nama Patung', 'Total (Rp)', 'Nama Artisan', 'Jumlah']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        print(f"✓ All required columns present")

        # Group by date and product
        try:
            print(f"📊 Grouping data by date and product...")
            df_grouped = df.groupby(['Tanggal', 'Nama Patung']).agg({
                'Total (Rp)': 'sum',
                'Jumlah': 'sum',
                'Nama Artisan': 'first',
                'Tour Guide': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None,
                'Metode Pembayaran': 'first'
            }).reset_index()
            print(f"✓ Grouped into {len(df_grouped)} transactions")
        except Exception as e:
            print(f"❌ Error grouping data: {e}")
            traceback.print_exc()
            return False

        # Check if database is connected
        if db is None:
            print(f"❌ Database not connected")
            return False

        # Seed transactions
        seeded_count = 0
        failed_count = 0
        
        print(f"💾 Seeding transactions to MongoDB...")
        
        for idx, row in df_grouped.iterrows():
            try:
                # Create SaleTransaction
                transaction_data = {
                    'receiptNumber': f"RCP{seeded_count:04d}",
                    'cashierId': None,
                    'guideId': None,
                    'guideNameSnapshot': row.get('Tour Guide') or '',
                    'guideCommissionRateSnapshot': 0,
                    'guideCommissionAmount': 0,
                    'subtotal': float(row['Total (Rp)']) if pd.notna(row['Total (Rp)']) else 0,
                    'discountAmount': 0,
                    'grandTotal': float(row['Total (Rp)']) if pd.notna(row['Total (Rp)']) else 0,
                    'paymentMethod': row.get('Metode Pembayaran') or 'Unknown',
                    'paidAmount': float(row['Total (Rp)']) if pd.notna(row['Total (Rp)']) else 0,
                    'changeAmount': 0,
                    'soldAt': pd.to_datetime(row['Tanggal'], errors='coerce')
                }
                
                # Validate transaction data
                if not isinstance(transaction_data['soldAt'], pd.Timestamp):
                    print(f"⚠ Warning: Invalid date for row {idx}, skipping")
                    continue

                transaction_result = db.saletransactions.insert_one(transaction_data)
                transaction_id = transaction_result.inserted_id

                # Create SaleItem
                item_data = {
                    'saleTransactionId': transaction_id,
                    'productItemId': None,
                    'masterProductId': None,
                    'artisanId': None,
                    'parentCodeSnapshot': row['Nama Patung'],
                    'childCodeSnapshot': f"{row['Nama Patung']}_001",
                    'categoryNameSnapshot': 'Patung',
                    'productNameSnapshot': row['Nama Patung'],
                    'artisanNameSnapshot': row['Nama Artisan'],
                    'costPriceSnapshot': float(row['Total (Rp)']) * 0.7 if pd.notna(row['Total (Rp)']) else 0,
                    'sellingPriceSnapshot': float(row['Total (Rp)']) if pd.notna(row['Total (Rp)']) else 0,
                    'quantity': int(row['Jumlah']) if pd.notna(row['Jumlah']) else 1,
                    'artisanCommissionRateSnapshot': 10,
                    'artisanCommissionAmount': float(row['Total (Rp)']) * 0.1 if pd.notna(row['Total (Rp)']) else 0
                }

                db.saleitems.insert_one(item_data)
                seeded_count += 1
                
                if (seeded_count % 100 == 0):
                    print(f"  ✓ Seeded {seeded_count} transactions...")
                    
            except Exception as e:
                failed_count += 1
                print(f"⚠ Warning: Error seeding row {idx}: {e}")
                continue

        print(f"=" * 60)
        print(f"📊 Seeding Summary")
        print(f"=" * 60)
        print(f"✓ Successfully seeded: {seeded_count} transactions")
        print(f"❌ Failed: {failed_count} transactions")
        print(f"Total: {len(df_grouped)} rows")
        print("=" * 60 + "\n")
        
        return seeded_count > 0
        
    except Exception as e:
        print(f"❌ Unexpected error in seed_data_from_excel: {e}")
        traceback.print_exc()
        return False

def main():
    """
    Main entry point for seed_data script.
    """
    print("=" * 60)
    print("🌱 KriyaLogic Data Seeding Engine")
    print("=" * 60 + "\n")
    
    # Connect to MongoDB
    global client, db
    client, db = connect_to_mongodb()
    
    if client is None or db is None:
        print("\n❌ Cannot proceed without database connection")
        sys.exit(1)
    
    try:
        # Run seeding
        success = seed_data_from_excel()
        
        if success:
            print("✓ Seeding completed successfully!")
            sys.exit(0)
        else:
            print("❌ Seeding failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Close database connection
        if client:
            client.close()
            print("✓ Database connection closed")

if __name__ == "__main__":
    main()