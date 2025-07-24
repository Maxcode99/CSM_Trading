import os
import re

def patch_pyltr():
    try:
        # Ruta base del proyecto (CSM_Trading/)
        base_path = os.path.dirname(__file__)
        venv_path = os.path.join(base_path, ".venv")
        file_path = os.path.join(venv_path, "Lib", "site-packages", "pyltr", "models", "lambdamart.py")

        if not os.path.exists(file_path):
            print("❌ No se encontró el archivo lambdamart.py")
            return

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        modified = False

        # Reemplazar np.object -> object
        if "np.object" in content:
            content = content.replace("np.object", "object")
            modified = True

        # Reemplazar np.bool -> bool
        if "np.bool" in content:
            content = content.replace("np.bool", "bool")
            modified = True

        # Eliminar argumento 'presort=...' dentro de cualquier DecisionTreeRegressor
        pattern = r"presort\s*=\s*[^,\)\n]+,?"
        if "DecisionTreeRegressor" in content and re.search(pattern, content):
            content = re.sub(pattern, "", content)
            modified = True

        # Guardar si hubo modificaciones
        if modified:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)
            print("✅ pyltr fue parchado correctamente.")
        else:
            print("ℹ️ pyltr ya estaba parchado.")

    except Exception as e:
        print("⚠️ Error al intentar parchear pyltr:", str(e))


# Ejecutar automáticamente al cargar el proyecto
patch_pyltr()
