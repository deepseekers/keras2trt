import tomlkit


def __get_project_meta():
    with open("./pyproject.toml") as pyproject:
        file_contents = pyproject.read()

    return tomlkit.parse(file_contents)["tool"]["poetry"]


def __get_package_version():
    pkg_meta = __get_project_meta()
    version = str(pkg_meta["version"])
    return version


__version__ = __get_package_version()
