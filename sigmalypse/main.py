import pandas as pd
from dataclasses import dataclass

@dataclass
class Measure:
    x: float  # Valor de la medición
    u: float  # Incertidumbre

    def __iter__(self):
        """Permite que la instancia se comporte como una tupla (x, u)."""
        return iter((self.x, self.u))

    def __neg__(self):
        """Devuelve una nueva instancia con el valor de x negativo."""
        return Measure(-self.x, self.u)

    def __repr__(self):
        """Representación en formato 'x ± u'."""
        return f"{self.x} ± {self.u}"

    def __getitem__(self, key):
        """Permite acceder a los valores como si fuera un diccionario."""
        if key == "x":
            return self.x
        elif key == "u":
            return self.u
        raise KeyError(f"Key {key} not found.")

    def __setitem__(self, key, value):
        """Permite modificar los valores como si fuera un diccionario."""
        if key == "x":
            self.x = value
        elif key == "u":
            self.u = value
        else:
            raise KeyError(f"Key {key} not found.")

@dataclass
class MeasureBOda(Measure):
    def __repr__(self):
        """Representación en formato 'x ± u'."""
        return f"{self.x} ± {self.u}"

    def __add__(self, other) -> 'MeasureBOda':
        """Suma dos medidas (incertidumbre absoluta). Si el otro operando es un entero o float, solo se opera el valor x; u se mantiene."""
        if isinstance(other, (int, float)):
            return MeasureBOda(self.x + other, self.u)
        elif isinstance(other, Measure):
            x_result = self.x + other.x
            u_result = self.u + other.u
            return MeasureBOda(x_result, u_result)
        else:
            raise TypeError("Addition is not supported between objects of different classes.")

    def __radd__(self, other) -> 'MeasureBOda':
        """Permite la suma con el escalar a la izquierda."""
        return self.__add__(other)

    def __sub__(self, other) -> 'MeasureBOda':
        """Resta dos medidas (incertidumbre absoluta). Si el otro operando es un entero o float, solo se opera el valor x; u se mantiene."""
        if isinstance(other, (int, float)):
            return MeasureBOda(self.x - other, self.u)
        elif isinstance(other, Measure):
            x_result = self.x - other.x
            u_result = self.u + other.u
            return MeasureBOda(x_result, u_result)
        else:
            raise TypeError("Subtraction is not supported between objects of different classes.")

    def __rsub__(self, other) -> 'MeasureBOda':
        """
        Permite la resta cuando el escalar está a la izquierda.
        Se calcula other - self.x y se mantiene la incertidumbre self.u.
        """
        if isinstance(other, (int, float)):
            return MeasureBOda(other - self.x, self.u)
        else:
            raise TypeError("Subtraction is not supported between objects of different classes.")

    def __mul__(self, other) -> 'MeasureBOda':
        """Multiplica dos medidas (incertidumbre absoluta). Si el otro operando es un entero o float, solo se opera el valor x; u se mantiene."""
        if isinstance(other, (int, float)):
            return MeasureBOda(self.x * other, self.u)
        elif isinstance(other, Measure):
            x_result = self.x * other.x
            u_result = self.x * other.u + self.u * other.x
            return MeasureBOda(x_result, u_result)
        else:
            raise TypeError("Multiplication is not supported between objects of different classes.")

    def __rmul__(self, other) -> 'MeasureBOda':
        """Permite la multiplicación cuando el escalar está a la izquierda."""
        return self.__mul__(other)

    def __truediv__(self, other) -> 'MeasureBOda':
        """Divide dos medidas (incertidumbre absoluta). Si el otro operando es un entero o float, solo se opera el valor x; u se mantiene."""
        if isinstance(other, (int, float)):
            return MeasureBOda(self.x / other, self.u)
        elif isinstance(other, Measure):
            x_result = self.x / other.x
            u_result = (self.x * other.u + self.u * other.x) / (other.x ** 2)
            return MeasureBOda(x_result, u_result)
        else:
            raise TypeError("Division is not supported between objects of different classes.")

    def __rtruediv__(self, other) -> 'MeasureBOda':
        """
        Permite la división cuando el escalar está a la izquierda.
        Se calcula other / self.x y se propaga la incertidumbre según la fórmula:
          u_result = (other * self.u) / (self.x ** 2)
        """
        if isinstance(other, (int, float)):
            return MeasureBOda(other / self.x, (other * self.u) / (self.x ** 2))
        else:
            raise TypeError("Division is not supported between objects of different classes.")

    def __pow__(self, power: float) -> 'MeasureBOda':
        """Eleva una medida a una potencia (incertidumbre absoluta)."""
        x_result = pow(self.x, power)
        u_result = abs(power) * pow(self.x, power - 1) * self.u
        return MeasureBOda(x_result, u_result)

    def round(self, c: int) -> 'MeasureBOda':
        return MeasureBOda(round(self.x, c), round(self.u, c))

class UnitCore:
    """
    Clase para conversión y verificación de unidades.

    Categorías:
      - L: Longitud (base = m)
      - M: Masa (base = kg)
      - T: Tiempo (base = s)
    """
    UNITS = {
        'L': {
            'km': 1000,  # 1 km = 1000 m
            'm': 1,
            'cm': 0.01,  # 1 cm = 0.01 m
            'mm': 0.001,  # 1 mm = 0.001 m
            'μm': 1e-6,  # 1 μm = 1e-6 m
            'nm': 1e-9  # 1 nm = 1e-9 m
        },
        'M': {
            'ton': 1000,  # 1 ton = 1000 kg
            'kg': 1,
            'g': 0.001,  # 1 g = 0.001 kg
            'mg': 1e-6,  # 1 mg = 1e-6 kg
            'μg': 1e-9  # 1 μg = 1e-9 kg
        },
        'T': {
            'hr': 3600,  # 1 hr = 3600 s
            'h': 3600,  # alias de hr
            'min': 60,  # 1 min = 60 s
            's': 1,
            'ms': 0.001,  # 1 ms = 0.001 s
            'μs': 1e-6,  # 1 μs = 1e-6 s
            'ns': 1e-9  # 1 ns = 1e-9 s
        }
    }

    @staticmethod
    def _parse_part(part: str) -> (float, dict):
        """
        Procesa una parte (numerador o denominador) de la unidad y retorna una
        tupla (factor_total, dim_total).
        """
        if not part:
            return (1, {})
        tokens = part.split("*")
        factor_total = 1
        dim_total = {}
        for token in tokens:
            if "^" in token:
                base, exp_str = token.split("^")
                try:
                    exp = float(exp_str)
                except ValueError:
                    raise ValueError(f"Exponente '{exp_str}' en la unidad '{token}' no es válido.")
            else:
                base = token
                exp = 1
            # Buscar la unidad base en UNITS y deducir su dimensión a partir de la categoría.
            found = False
            for cat_key, cat_units in UnitCore.UNITS.items():
                if base in cat_units:
                    base_factor = cat_units[base]
                    base_dim = {cat_key: 1}  # La dimensión se deduce de la categoría.
                    found = True
                    break
            if not found:
                raise ValueError(f"Unidad base '{base}' no soportada.")
            factor_total *= base_factor ** exp
            # Acumular la dimensión multiplicando el vector por el exponente.
            for key, value in base_dim.items():
                dim_total[key] = dim_total.get(key, 0) + value * exp
        return (factor_total, dim_total)

    @staticmethod
    def _parse_unit_components(unit: str) -> (float, dict):
        """
        Separa la cadena de unidad en numerador y denominador, procesa cada parte y
        retorna una tupla (net_factor, net_dim).
        """
        unit = unit.replace(" ", "")
        if unit == "":
            return (1, {})  # Unidad vacía se interpreta como adimensional.

        if "/" in unit:
            num_str, den_str = unit.split("/", 1)
        else:
            num_str = unit
            den_str = ""

        num_factor, num_dim = UnitCore._parse_part(num_str)
        den_factor, den_dim = UnitCore._parse_part(den_str)
        net_factor = num_factor / den_factor

        # Combinar dimensiones: restar la dimensión del denominador a la del numerador.
        net_dim = {}
        for key in set(num_dim.keys()).union(den_dim.keys()):
            net_dim[key] = num_dim.get(key, 0) - den_dim.get(key, 0)
        return (net_factor, net_dim)

    @staticmethod
    def analyze_dimension(unit: str) -> dict:
        """
        Retorna el vector dimensional de la unidad (simple o compuesta).
        """
        _, net_dim = UnitCore._parse_unit_components(unit)
        return net_dim

    @staticmethod
    def extract_factor(unit: str) -> float:
        """
        Retorna el factor de conversión de la unidad (simple o compuesta).
        """
        net_factor, _ = UnitCore._parse_unit_components(unit)
        return net_factor

    @staticmethod
    def check_unit(unit: str) -> (float, dict):
        """
        Verifica la unidad y retorna una tupla (factor, dimensión).
        Esta función utiliza analyze_dimension y extract_factor.
        """
        factor = UnitCore.extract_factor(unit)
        dim = UnitCore.analyze_dimension(unit)
        return factor, dim

    @staticmethod
    def unit_conversion(v: float, unit_in: str, unit_out: str) -> float:
        """
        Convierte un valor entre dos unidades simples (dentro de la misma categoría),
        utilizando UNITS.
        """
        for cat_key, cat_units in UnitCore.UNITS.items():
            if unit_in in cat_units and unit_out in cat_units:
                return v * cat_units[unit_in] / cat_units[unit_out]
        raise ValueError(
            f"Las unidades '{unit_in}' y '{unit_out}' no pertenecen a la misma categoría o no son válidas.")

    @staticmethod
    def composite_conversion(v: float, unit_in: str, unit_out: str) -> float:
        """
        Convierte un valor entre unidades compuestas, utilizando check_unit para obtener
        el factor y el vector dimensional. Verifica que las unidades de entrada y salida sean
        dimensionalmente equivalentes.
        """
        in_factor, in_dim = UnitCore.check_unit(unit_in)
        out_factor, out_dim = UnitCore.check_unit(unit_out)
        if in_dim != out_dim:
            raise ValueError(
                f"Incompatibilidad dimensional: '{unit_in}' tiene dimensiones {in_dim} y "
                f"'{unit_out}' tiene dimensiones {out_dim}."
            )
        return v * in_factor / out_factor

class MetricsLab:
    def __init__(self, units=None):
        # Unidades de trabajo: si no se especifica, se usan las estándar (base: m, kg, s)
        self.units = units or {"L": "m", "M": "kg", "T": "s"}
        self._validate_units()
        # Valores por defecto de incertidumbre en las unidades base
        self.default_u_objects = {
            "L": {  # Longitud (en metros)
                "vernier": 5e-5,  # 0.00005 m
                "flexometro": 0.005,  # 0.005 m
                "micrometro": 1e-5  # 0.00001 m
            },
            "M": {  # Masa (en kilogramos)
                "balanza digital": 0.00005,  # 0.00005 kg (0.05 g)
                "balanza analítica": 1e-4,  # 0.0001 kg (0.1 g)
                "balanza granataria": 0.0005  # 0.0005 kg (0.5 g)
            },
            "T": {  # Tiempo (en segundos)
                "cronometro": 0.01,  # 0.001 s
                "fotocompuerta": 0.0001  # 0.0001 s
            }
        }
        # Actualiza las incertidumbres según las unidades de trabajo
        self.update_instrument_uncertainties()
        self.tables = {}

    def _validate_units(self):
        """
        Valida que el diccionario self.units tenga claves válidas (según UnitCore.UNITS)
        y que cada unidad asignada pertenezca a la lista de unidades permitidas para esa categoría.
        """
        for category, unit in self.units.items():
            if category not in UnitCore.UNITS:
                raise ValueError(
                    f"Categoría '{category}' no es válida. Debe ser una de: {list(UnitCore.UNITS.keys())}."
                )
            if unit not in UnitCore.UNITS[category]:
                raise ValueError(
                    f"Unidad '{unit}' no es válida para la categoría '{category}'. "
                    f"Unidades válidas: {list(UnitCore.UNITS[category].keys())}."
                )

    def update_instrument_uncertainties(self):
        """
        Actualiza las incertidumbres de los instrumentos transformándolas desde las unidades base
        a las unidades de trabajo definidas en self.units.
        """
        base_units = {"L": "m", "M": "kg", "T": "s"}
        self.U_OBJECTS = {}
        for cat, instruments in self.default_u_objects.items():
            working_unit = self.units.get(cat)
            if working_unit is None:
                raise ValueError(f"No se ha definido una unidad de trabajo para la categoría '{cat}'.")
            factor = UnitCore.unit_conversion(1, base_units[cat], working_unit)
            self.U_OBJECTS[cat] = {instr: unc * factor for instr, unc in instruments.items()}

    def add_table_from_csv(self, src: str, name: str, units: dict):
        """
        Lee un archivo CSV usando pandas, valida las unidades de cada columna y añade la tabla
        en el diccionario 'tables' con el nombre especificado.

        Parámetros:
            src (str): Ruta del archivo CSV.
            name (str): Nombre con el que se almacenará la tabla.
            units (dict): Diccionario con la estructura {"nombre_columna": "unidad de medida"}.
                           Si la unidad es "text", no se realiza validación; de lo contrario, se valida
                           usando UnitCore.check_unit.
        """
        try:
            df = pd.read_csv(src)
        except Exception as e:
            print(f"Error al leer el archivo CSV: {e}")
            return

        for col, unit in units.items():
            if unit.lower() != "text":
                try:
                    UnitCore.check_unit(unit)
                except Exception as e:
                    raise ValueError(f"Unidad inválida para la columna '{col}': {unit}. Error: {e}")

        self.tables[name] = {"data": df, "units": units}
        print(f"Tabla '{name}' añadida exitosamente.")

    def add_uncertainty(self, table: str, col: str, obj: str):
        """
        Agrega la incertidumbre de un instrumento a los valores de una columna en una tabla.

        Parámetros:
            table (str): Nombre de la tabla en self.tables.
            col (str): Nombre de la columna a la que se le añadirá la incertidumbre.
            obj (str): Nombre del instrumento (por ejemplo, "vernier" o "balanza digital") para obtener su incertidumbre.

        Para cada valor en la columna, se crea una instancia de MeasureBOda con el valor y la incertidumbre.
        """
        instrument_unc = None
        for category, instruments in self.U_OBJECTS.items():
            if obj in instruments:
                instrument_unc = instruments[obj]
                break
        if instrument_unc is None:
            raise ValueError(f"Instrumento '{obj}' no encontrado en U_OBJECTS.")

        if table not in self.tables:
            raise ValueError(f"La tabla '{table}' no existe.")
        df = self.tables[table]["data"]
        if col not in df.columns:
            raise ValueError(f"La columna '{col}' no existe en la tabla '{table}'.")

        df[col] = df[col].apply(lambda x: MeasureBOda(x, instrument_unc))
        print(f"Incertidumbre agregada a la columna '{col}' de la tabla '{table}' usando el instrumento '{obj}'.")

    def convert_table_units(self, table: str):
        """
        Convierte las unidades de medida de todas las columnas de la tabla especificada a las unidades de trabajo
        definidas en self.units. Para cada columna (excepto aquellas con unidad "text"), se determina su categoría (L, M o T)
        usando UnitCore.UNITS o mediante análisis dimensional; se obtiene la unidad de trabajo para esa categoría y se
        convierte cada valor.

        Si el valor es numérico o es una instancia de Measure (o MeasureBOda), se aplica la conversión correspondiente.
        Finalmente, se actualiza la unidad de la columna en el diccionario "units" de la tabla.
        """
        if table not in self.tables:
            raise ValueError(f"La tabla '{table}' no existe.")
        df = self.tables[table]["data"]
        units_dict = self.tables[table]["units"]

        for col, orig_unit in units_dict.items():
            if orig_unit.lower() == "text":
                continue

            # Determinar la categoría de la unidad.
            category = None
            for cat_key, cat_units in UnitCore.UNITS.items():
                if orig_unit in cat_units:
                    category = cat_key
                    break
            # Si no se encuentra de forma simple, intentar deducir mediante análisis dimensional.
            if category is None:
                _, dim = UnitCore.check_unit(orig_unit)
                keys = [k for k, v in dim.items() if v != 0]
                if len(keys) == 1:
                    category = keys[0]
            if category is None:
                raise ValueError(
                    f"No se pudo determinar la categoría de la unidad '{orig_unit}' en la columna '{col}'.")

            working_unit = self.units.get(category)
            if working_unit is None:
                raise ValueError(f"No se ha definido una unidad de trabajo para la categoría '{category}'.")
            if orig_unit == working_unit:
                continue

            # Elegir el método de conversión: si la unidad es compuesta, se usa composite_conversion; si no, unit_conversion.
            if any(ch in orig_unit for ch in ["/", "*", "^"]):
                factor = UnitCore.composite_conversion(1, orig_unit, working_unit)
            else:
                factor = UnitCore.unit_conversion(1, orig_unit, working_unit)

            def convert_value(x):
                if isinstance(x, Measure):
                    return MeasureBOda(x.x * factor, x.u * factor)
                else:
                    return x * factor

            df[col] = df[col].apply(convert_value)
            units_dict[col] = working_unit

        print(f"Unidades convertidas en la tabla '{table}'.")

    def __repr__(self):
        return f"MetricsLab:\n - Units: {self.units}\n - Tables: {list(self.tables.keys())}"


# if __name__ == "__main__":
#     lab = MetricsLab(units={"L": "cm", "M": "g", "T": "s"})
#     lab.add_table_from_csv(src="monedas.csv", name="test", units={"diametro": "cm", "altura": "cm", "masa": "g"})
#     lab.convert_table_units("monedas")
#     lab.add_uncertainty("monedas", "diametro", "vernier")
#     print(lab.tables["test"])
