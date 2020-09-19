import typing as t

from IPython.testing.globalipapp import start_ipython
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def session_ip():
    yield start_ipython()


@pytest.fixture(scope="function")
def ip(session_ip):
    session_ip.run_line_magic(magic_name="load_ext", line="pandacell")
    yield session_ip
    session_ip.run_line_magic(magic_name="unload_ext", line="pandacell")
    session_ip.run_line_magic(magic_name="reset", line="-f")


def test_no_dataframe(ip):
    with pytest.raises(NameError):
        ip.run_line_magic("df", "")


def test_assign_variable(ip):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0])")
    res = ip.run_line_magic("df", "a=1")
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    pd.testing.assert_frame_equal(res, exp)


def test_assign_two_variables(ip):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0])")
    res = ip.run_cell_magic("df", line="", cell="a=1\nb=2")
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    exp["b"] = 2
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("flag", ["-i", "--inplace"])
def test_assign_variable_inplace(ip, flag: str):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0])")
    ip.run_line_magic("df", f"{flag} a=1")
    res = ip.user_global_ns["df"]
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("flag", ["-i", "--inplace"])
def test_assign_multiple_variables_inplace(ip, flag: str):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0])")
    ip.run_cell_magic("df", line=flag, cell="a=1\nb=2")
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    exp["b"] = 2
    res = ip.user_global_ns["df"]
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("flag", ["-n", "--name"])
def test_other_name(ip, flag: str):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df2 = pd.DataFrame(index=[0])")
    res = ip.run_line_magic("df", f"{flag}=df2 a=1")
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("flag", ["-q", "--query"])
def test_query(ip, flag: str):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0,1], data={'a': [1,2]})")
    res = ip.run_line_magic("df", f"{flag} a==1")
    exp = pd.DataFrame(index=[0])
    exp["a"] = 1
    pd.testing.assert_frame_equal(res, exp)


@pytest.mark.parametrize("flag", ["-q", "--query"])
def test_query_multiline_fails(ip, flag: str):
    ip.run_cell("import pandas as pd")
    ip.run_cell("df = pd.DataFrame(index=[0,1], data={'a': [1,2]})")
    with pytest.raises(ValueError):
        ip.run_cell_magic("df", line=flag, cell="a==1\nb==2")


@pytest.mark.parametrize("multiline", [True, False])
@pytest.mark.parametrize("inplace", ["-i", "--inplace", None])
@pytest.mark.parametrize("name", ["-n=df2", "--name=df2", None])
@pytest.mark.parametrize("query", ["-q", "--query", None])
def test_flag_combinations(
    ip,
    multiline: bool,
    inplace: t.Optional[str],
    name: t.Optional[str],
    query: t.Optional[str],
):
    if query and multiline:
        return
    ip.run_cell("import pandas as pd")
    df_name = "df" if not name else "df2"
    ip.run_cell(f"{df_name} = pd.DataFrame(index=[0,1], data={{'a': [1,2]}})")
    exec_string = "a=1" if not query else "a==1"
    if multiline:
        exec_string += "\n"
        exec_string += exec_string.replace("a", "b")
    flags = " ".join([opt for opt in (inplace, name, query) if opt is not None])
    if not multiline:
        res = ip.run_line_magic("df", f"{flags} {exec_string}")
    else:
        res = ip.run_cell_magic("df", line=flags, cell=exec_string)
    if inplace:
        res = ip.user_global_ns[df_name]
    index = [0, 1] if not query else [0]
    exp = pd.DataFrame(index=index)
    exp["a"] = 1
    if multiline:
        exp["b"] = 1
    pd.testing.assert_frame_equal(res, exp)
