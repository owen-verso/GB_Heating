from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 0)
pd.set_option("display.expand_frame_repr", False)

# Constant for CO2 intensity of gas
GAS_CARBON_INTENSITY = 0.235  # kgCO2/kWh

class Scenario:
    """
    A scenario for gas, electricity, subsidy, and CO2 tax.
    param: gas_price in £/kWh
    param: elec_price in £/kWh
    param: co2_tax_t in £/t CO2 emitted
    param: subsidy in £/heat pump installation
    """
    def __init__(self, label="Scenario 1", gas_price=6.24, elec_price=24.5, co2_tax_t=0.0, subsidy=7500):
        self.label = label
        self.gas_price_pretax = gas_price
        self.elec_price = elec_price
        self.co2_tax_kg = co2_tax_t/1000
        self.subsidy = subsidy
        self.gas_price_posttax = self.gas_price_pretax + self.co2_tax_kg * GAS_CARBON_INTENSITY

    def return_df(self):
        return pd.DataFrame({
            'elec_price_recent': [self.elec_price],
            'gas_price_recent': [self.gas_price_posttax],
            'subsidy_level': [self.subsidy]
        }, index=[0])

class HeatPumpUptakeModel:
    """
    A linear regression model for heat pump installations.
    """
    def __init__(self, filename):
        self.df = pd.read_excel(filename)
        self._prepare_data()
        self._fit_model()

    def _prepare_data(self):
        df = self.df.copy()
        df['elec_price_recent'] = df[['elec_price', 'elec_price_lag1', 'elec_price_lag2']].mean(axis=1)
        df['gas_price_recent'] = df[['gas_price', 'gas_price_lag1', 'gas_price_lag2']].mean(axis=1)
        df.drop(['elec_price', 'gas_price', 'elec_price_lag1', 'elec_price_lag2',
                 'gas_price_lag1', 'gas_price_lag2'], axis=1, inplace=True)
        df = df[13:]
        df.dropna(subset=['elec_price_recent', 'gas_price_recent', 'subsidy_level', 'installations'], inplace=True)
        self.df = df
        self.features = ['elec_price_recent', 'gas_price_recent', 'subsidy_level']
        self.target = 'installations'

    def _fit_model(self):
        X = self.df[self.features]
        y = self.df[self.target]
        self.model = LinearRegression()
        self.model.fit(X, y)
        y_pred = self.model.predict(X)
        r2 = r2_score(y, y_pred)
        print("Regression Coefficients:")
        # Coefficients and summary
        coefficients = pd.DataFrame({
            'Feature': self.features,
            'Coefficient': self.model.coef_
        }).sort_values(by='Coefficient', key=abs, ascending=False)
        print(coefficients)
        print(f"R² score: {r2:.3f}")

    def scenario_projection(self, scenario: Scenario):
        prediction = self.model.predict(scenario.return_df())[0]
        return {
            "Projected Quarterly Installations": round(prediction),
            "Projected Yearly Installations": round(prediction * 4) * 2 + 250000
        }#Change from quarterly projections to yearly, multiply by 2 to adjust to FES modelled values, add 250k/year of new builds

# Create Model
model = HeatPumpUptakeModel("gb_heat_V2.xlsx")

# Scenarios tested:
scenarios = [
    Scenario(label='Standard Scenario', co2_tax_t=62),
    Scenario(label='Standard Scenario high CO2 tax', co2_tax_t=103),
    Scenario(label='Standard High Elec', elec_price=40, co2_tax_t=62),
    Scenario(label='Standard Low Elec', elec_price=15, co2_tax_t=62),
    Scenario(label='Standard High Gas', gas_price=15,co2_tax_t=62),
    Scenario(label='Standard Low Gas', gas_price=4, co2_tax_t=62),
    Scenario(label='Standard Subsidy High Gas High CO2 Tax Low Elec', gas_price=15, elec_price=15, co2_tax_t=103),
    Scenario(label='Standard Low Gas Low CO2 Tax High Elec', gas_price=4, elec_price=40, co2_tax_t=40),
    Scenario(label='Increased Subsidy', co2_tax_t=62, subsidy=10000),
    Scenario(label='Increased Subsidy High Elec', elec_price=40, co2_tax_t=62, subsidy=10000),
    Scenario(label='Increased Subsidy Low Elec', elec_price=15, co2_tax_t=62, subsidy=10000),
    Scenario(label='Increased Subsidy High Gas', gas_price=15, co2_tax_t=62, subsidy=10000),
    Scenario(label='Increased Subsidy Low Gas', gas_price=4, co2_tax_t=62, subsidy=10000),
    Scenario(label='Increased Subsidy Low Elec High CO2 Tax', elec_price=15, co2_tax_t=103, subsidy=10000),
    Scenario(label='Increased Subsidy High Gas High CO2 Tax', gas_price=15, co2_tax_t=103, subsidy=10000),
    Scenario(label='Increased Subsidy High Gas High CO2 Tax Low Elec', gas_price=15, elec_price=15, co2_tax_t=103, subsidy=10000),
    Scenario(label='Big Subsidy High Gas High CO2 Tax Low Elec', gas_price=15, elec_price=15, co2_tax_t=103,
             subsidy=15000),
    Scenario(label='Increased Subsidy Low Gas Low CO2 Tax High Elec', gas_price=4, elec_price=40, co2_tax_t=40, subsidy=10000)
]

# Format projection results
projections = []

for sc in scenarios:
    result = model.scenario_projection(sc)
    result['Scenario'] = sc.label
    result['Gas price (post-tax)'] = sc.gas_price_posttax
    result['Electricity price'] = sc.elec_price
    result['Subsidy (£)'] = sc.subsidy
    result['CO₂ tax (£/tCO₂)'] = sc.co2_tax_kg * 1000  # convert back to £/t for display
    projections.append(result)


projection_df = pd.DataFrame(projections)

# Display projection results
pd.set_option("display.float_format", "{:,.0f}".format)
print(projection_df[['Scenario', 'Gas price (post-tax)', 'Electricity price', 'Subsidy (£)', 'CO₂ tax (£/tCO₂)',
                      'Projected Yearly Installations']])
