#!/usr/bin/env python3
"""
Medicaid Data Analysis Guide
Complete guide for analyzing downloaded Medicaid data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
sns.set_palette("husl")


class MedicaidDataAnalyzer:
    """
    Comprehensive analysis toolkit for Medicaid data
    """

    def __init__(self, data_file_path: str):
        """
        Initialize with your downloaded data file

        Args:
            data_file_path: Path to your downloaded CSV/Excel file
        """
        self.data_file_path = data_file_path
        self.df = None
        self.load_data()

    def load_data(self):
        """Load the data from file"""
        print(f"📂 Loading data from: {self.data_file_path}")

        try:
            if self.data_file_path.endswith('.csv'):
                self.df = pd.read_csv(self.data_file_path)
            elif self.data_file_path.endswith(('.xlsx', '.xls')):
                self.df = pd.read_excel(self.data_file_path)
            elif self.data_file_path.endswith('.json'):
                self.df = pd.read_json(self.data_file_path)
            else:
                # Try CSV as default
                self.df = pd.read_csv(self.data_file_path)

            print(f"✅ Successfully loaded {len(self.df):,} rows and {len(self.df.columns)} columns")
            print(f"💾 Memory usage: {self.df.memory_usage(deep=True).sum() / 1024 ** 2:.1f} MB")

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None

    def explore_data_structure(self):
        """
        Explore the basic structure of your data
        """
        print("\n" + "=" * 60)
        print("🔍 DATA STRUCTURE EXPLORATION")
        print("=" * 60)

        # Basic info
        print(f"📊 Dataset shape: {self.df.shape}")
        print(f"🗓️ Data loaded on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Column information
        print(f"\n📑 Columns ({len(self.df.columns)}):")
        for i, col in enumerate(self.df.columns, 1):
            dtype = self.df[col].dtype
            null_count = self.df[col].isnull().sum()
            null_pct = (null_count / len(self.df)) * 100
            unique_vals = self.df[col].nunique()

            print(
                f"  {i:2d}. {col:<30} | {str(dtype):<10} | {null_count:>6} nulls ({null_pct:4.1f}%) | {unique_vals:>6} unique")

        # Data types summary
        print(f"\n🏷️ Data types summary:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"  {dtype}: {count} columns")

        # Memory usage by column
        print(f"\n💾 Top 5 memory-consuming columns:")
        memory_usage = self.df.memory_usage(deep=True).sort_values(ascending=False)
        for col, usage in memory_usage.head(6).items():  # 6 because index is included
            if col != 'Index':
                print(f"  {col:<30}: {usage / 1024 ** 2:>6.1f} MB")

    def data_quality_check(self):
        """
        Check data quality issues
        """
        print("\n" + "=" * 60)
        print("🔎 DATA QUALITY CHECK")
        print("=" * 60)

        # Missing values analysis
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)

        if len(missing_data) > 0:
            print(f"❌ Columns with missing values:")
            for col, count in missing_data.items():
                pct = (count / len(self.df)) * 100
                print(f"  {col:<30}: {count:>8,} ({pct:5.1f}%)")
        else:
            print("✅ No missing values found!")

        # Duplicate rows
        duplicate_count = self.df.duplicated().sum()
        print(f"\n🔄 Duplicate rows: {duplicate_count:,}")

        # Data consistency checks
        print(f"\n🔍 Data consistency checks:")

        # Check for columns that might be dates
        potential_date_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month']):
                potential_date_cols.append(col)

        if potential_date_cols:
            print(f"  📅 Potential date columns: {', '.join(potential_date_cols)}")

        # Check for columns with very few unique values (potential categories)
        low_cardinality = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05:  # Less than 5% unique values
                    low_cardinality.append((col, self.df[col].nunique()))

        if low_cardinality:
            print(f"  🏷️ Low cardinality columns (potential categories):")
            for col, unique_count in low_cardinality:
                print(f"    {col}: {unique_count} unique values")

    def statistical_summary(self):
        """
        Generate statistical summary
        """
        print("\n" + "=" * 60)
        print("📈 STATISTICAL SUMMARY")
        print("=" * 60)

        # Numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(f"🔢 Numeric columns summary:")
            print(self.df[numeric_cols].describe().round(2))

        # Categorical columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"\n📝 Categorical columns summary:")
            for col in categorical_cols[:5]:  # Show first 5 categorical columns
                print(f"\n  {col}:")
                value_counts = self.df[col].value_counts().head(10)
                for value, count in value_counts.items():
                    pct = (count / len(self.df)) * 100
                    print(f"    {str(value):<20}: {count:>8,} ({pct:4.1f}%)")

                if self.df[col].nunique() > 10:
                    print(f"    ... and {self.df[col].nunique() - 10} more unique values")

    def find_key_insights(self):
        """
        Automatically find key insights in the data
        """
        print("\n" + "=" * 60)
        print("💡 KEY INSIGHTS")
        print("=" * 60)

        insights = []

        # Time-based insights
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'year']):
                try:
                    if 'year' in col.lower():
                        year_range = f"{self.df[col].min()} to {self.df[col].max()}"
                        insights.append(f"📅 Data spans years: {year_range}")
                    else:
                        # Try to parse as date
                        date_series = pd.to_datetime(self.df[col], errors='coerce')
                        if not date_series.isnull().all():
                            date_range = f"{date_series.min().strftime('%Y-%m-%d')} to {date_series.max().strftime('%Y-%m-%d')}"
                            insights.append(f"📅 Date range: {date_range}")
                except:
                    continue

        # Geographic insights
        geo_keywords = ['state', 'county', 'city', 'zip', 'region']
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in geo_keywords):
                unique_locations = self.df[col].nunique()
                insights.append(f"🗺️ Geographic coverage: {unique_locations} unique {col.lower()}s")

                # Show top locations
                top_locations = self.df[col].value_counts().head(3)
                top_list = ', '.join([f"{loc} ({count:,})" for loc, count in top_locations.items()])
                insights.append(f"   Top locations: {top_list}")
                break

        # Large number insights
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].max() > 1000000:  # Large numbers
                max_val = self.df[col].max()
                insights.append(
                    f"💰 Largest {col}: ${max_val:,.0f}" if 'cost' in col.lower() or 'amount' in col.lower() else f"🔢 Largest {col}: {max_val:,.0f}")

        # Print insights
        for insight in insights[:10]:  # Limit to top 10 insights
            print(f"  {insight}")

        if not insights:
            print("  🤔 Run specific analysis functions to discover insights!")

    def create_visualizations(self):
        """
        Create useful visualizations
        """
        print("\n" + "=" * 60)
        print("📊 CREATING VISUALIZATIONS")
        print("=" * 60)

        # Set up the plotting area
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Medicaid Data Overview', fontsize=16, fontweight='bold')

        # 1. Missing data heatmap
        ax1 = axes[0, 0]
        missing_data = self.df.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0].sort_values()
            missing_data.plot(kind='barh', ax=ax1, color='coral')
            ax1.set_title('Missing Data by Column')
            ax1.set_xlabel('Number of Missing Values')
        else:
            ax1.text(0.5, 0.5, 'No Missing Data!', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
            ax1.set_title('Missing Data Check')

        # 2. Data types distribution
        ax2 = axes[0, 1]
        dtype_counts = self.df.dtypes.value_counts()
        colors = plt.cm.Set3(np.linspace(0, 1, len(dtype_counts)))
        ax2.pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%', colors=colors)
        ax2.set_title('Data Types Distribution')

        # 3. Numeric data distribution (first numeric column)
        ax3 = axes[1, 0]
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            self.df[first_numeric].hist(bins=30, ax=ax3, alpha=0.7, color='skyblue')
            ax3.set_title(f'Distribution of {first_numeric}')
            ax3.set_xlabel(first_numeric)
            ax3.set_ylabel('Frequency')
        else:
            ax3.text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Numeric Data Distribution')

        # 4. Top categories (first categorical column)
        ax4 = axes[1, 1]
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            first_categorical = categorical_cols[0]
            top_categories = self.df[first_categorical].value_counts().head(10)
            top_categories.plot(kind='bar', ax=ax4, color='lightgreen')
            ax4.set_title(f'Top 10 {first_categorical}')
            ax4.set_xlabel(first_categorical)
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        else:
            ax4.text(0.5, 0.5, 'No Categorical Columns', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Categorical Data')

        plt.tight_layout()
        plt.savefig('medicaid_data_overview.png', dpi=300, bbox_inches='tight')
        print("📊 Saved visualization as 'medicaid_data_overview.png'")
        plt.show()

    def advanced_analysis_examples(self):
        """
        Show examples of advanced analysis you can do
        """
        print("\n" + "=" * 60)
        print("🎓 ADVANCED ANALYSIS EXAMPLES")
        print("=" * 60)

        print("Here are some advanced analyses you can perform:")

        # Example 1: Time series analysis
        print("\n1️⃣ TIME SERIES ANALYSIS:")
        print("   # Group by time period")
        print("   monthly_trends = df.groupby('year_month').agg({")
        print("       'beneficiaries': 'sum',")
        print("       'total_cost': 'mean'")
        print("   })")
        print("   monthly_trends.plot()")

        # Example 2: Geographic analysis
        print("\n2️⃣ GEOGRAPHIC ANALYSIS:")
        print("   # Compare by state/region")
        print("   state_comparison = df.groupby('state').agg({")
        print("       'total_spending': 'sum',")
        print("       'beneficiaries': 'count'")
        print("   })")
        print(
            "   state_comparison['per_capita'] = state_comparison['total_spending'] / state_comparison['beneficiaries']")

        # Example 3: Correlation analysis
        print("\n3️⃣ CORRELATION ANALYSIS:")
        print("   # Find relationships between numeric variables")
        print("   correlation_matrix = df.select_dtypes(include=[np.number]).corr()")
        print("   sns.heatmap(correlation_matrix, annot=True)")

        # Example 4: Filtering and segmentation
        print("\n4️⃣ DATA FILTERING & SEGMENTATION:")
        print("   # Filter for specific criteria")
        print("   high_cost_programs = df[df['total_cost'] > df['total_cost'].quantile(0.9)]")
        print("   recent_data = df[df['year'] >= 2020]")

        # Example 5: Statistical tests
        print("\n5️⃣ STATISTICAL TESTS:")
        print("   # Compare groups")
        print("   from scipy import stats")
        print("   group1 = df[df['program_type'] == 'A']['cost_per_beneficiary']")
        print("   group2 = df[df['program_type'] == 'B']['cost_per_beneficiary']")
        print("   t_stat, p_value = stats.ttest_ind(group1, group2)")

    def export_summary_report(self):
        """
        Export a summary report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"medicaid_data_report_{timestamp}.txt"

        with open(report_filename, 'w') as f:
            f.write("MEDICAID DATA ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data file: {self.data_file_path}\n")
            f.write(f"Records: {len(self.df):,}\n")
            f.write(f"Columns: {len(self.df.columns)}\n\n")

            f.write("COLUMN SUMMARY:\n")
            for col in self.df.columns:
                f.write(f"- {col}: {self.df[col].dtype}, {self.df[col].nunique()} unique values\n")

            f.write(f"\nMISSING DATA:\n")
            missing = self.df.isnull().sum()
            missing = missing[missing > 0]
            if len(missing) > 0:
                for col, count in missing.items():
                    f.write(f"- {col}: {count} missing ({count / len(self.df) * 100:.1f}%)\n")
            else:
                f.write("- No missing data\n")

        print(f"📄 Summary report saved as: {report_filename}")


# Example usage functions
def quick_analysis(file_path: str):
    """
    Quick analysis of your Medicaid data
    """
    analyzer = MedicaidDataAnalyzer(file_path)

    if analyzer.df is not None:
        analyzer.explore_data_structure()
        analyzer.data_quality_check()
        analyzer.statistical_summary()
        analyzer.find_key_insights()
        analyzer.create_visualizations()
        analyzer.export_summary_report()

    return analyzer


def custom_analysis_examples(df):
    """
    Examples of custom analyses you can do with your data
    """
    print("\n" + "=" * 60)
    print("🔧 CUSTOM ANALYSIS EXAMPLES")
    print("=" * 60)

    # Example: Find top 10 records by a numeric column
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        first_numeric = numeric_cols[0]
        print(f"\n📊 Top 10 records by {first_numeric}:")
        top_10 = df.nlargest(10, first_numeric)
        print(top_10.to_string())

    # Example: Group by categorical column
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        first_cat = categorical_cols[0]
        first_num = numeric_cols[0]

        print(f"\n📈 Average {first_num} by {first_cat}:")
        grouped = df.groupby(first_cat)[first_num].agg(['mean', 'count', 'sum']).round(2)
        grouped = grouped.sort_values('mean', ascending=False).head(10)
        print(grouped.to_string())

    # Example: Create percentage calculations
    if len(categorical_cols) > 0:
        first_cat = categorical_cols[0]
        print(f"\n📊 Percentage distribution of {first_cat}:")
        percentages = df[first_cat].value_counts(normalize=True) * 100
        for value, pct in percentages.head(10).items():
            print(f"  {value}: {pct:.1f}%")


def json_analysis_workflow(file_path: str):
    """
    Specialized workflow for JSON data analysis
    """
    print("🏥 MEDICAID JSON DATA ANALYSIS")
    print("=" * 50)

    analyzer = MedicaidDataAnalyzer(file_path)

    if analyzer.df is None:
        print("❌ Failed to load data. Let's inspect the JSON structure:")
        analyzer.inspect_json_structure()
        return None

    # Handle nested JSON columns
    analyzer.handle_nested_json_columns()

    # Continue with standard analysis
    analyzer.explore_data_structure()
    analyzer.data_quality_check()
    analyzer.statistical_summary()
    analyzer.find_key_insights()
    analyzer.create_visualizations()
    analyzer.export_summary_report()

    return analyzer


def inspect_json_before_loading(file_path: str):
    """
    Just inspect JSON structure without full analysis
    """
    temp_analyzer = MedicaidDataAnalyzer.__new__(MedicaidDataAnalyzer)
    temp_analyzer.data_file_path = file_path
    temp_analyzer.inspect_json_structure()


if __name__ == "__main__":
    # Example usage - replace with your actual file path
    data_file = "/Users/emilyquick-cole/Downloads/medicaid_data/medicaid_dataset_20250905_191408.json"  # JSON file

    print("🏥 MEDICAID DATA ANALYSIS TOOLKIT")
    print("=" * 50)

    try:
        # For JSON files, you can inspect structure first
        if data_file.endswith('.json'):
            print("🔍 First, let's inspect the JSON structure:")
            inspect_json_before_loading(data_file)

            print("\n" + "=" * 50)
            print("📊 Now running full analysis:")
            analyzer = json_analysis_workflow(data_file)
        else:
            # Regular analysis for CSV/Excel
            analyzer = quick_analysis(data_file)

        # Custom analysis examples
        if analyzer and analyzer.df is not None:
            custom_analysis_examples(analyzer.df)

        print("\n✅ Analysis complete!")
        print("💡 Check the generated files:")
        print("   - medicaid_data_overview.png (visualizations)")
        print("   - medicaid_data_report_*.txt (summary report)")

    except FileNotFoundError:
        print("❌ File not found!")
        print("💡 Make sure to:")
        print("   1. Update the 'data_file' variable with your actual file path")
        print("   2. Run the downloader script first to get the data")
        print("\n🔍 If you want to just inspect a JSON file structure:")
        print("   inspect_json_before_loading('path/to/your/file.json')")

    except Exception as e:
        print(f"❌ Error: {e}")