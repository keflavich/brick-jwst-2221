"""
https://stackoverflow.com/questions/67811531/how-can-i-execute-a-ipynb-notebook-file-in-a-python-script
https://nbconvert.readthedocs.io/en/latest/execute_api.html
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def run_notebook(filename, kernel_name='python39'):
    print(f"Running notebook {filename}")
    with open(filename) as ff:
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)

    print(f"Writing backup notebook {filename}.backup")
    with open(filename + ".backup", 'w', encoding='utf-8') as fh:
        nbformat.write(nb_in, fh)

    ep = ExecutePreprocessor(timeout=1200)

    nb_out = ep.preprocess(nb_in)

    print(f"Writing notebook {filename}")
    with open(filename, 'w', encoding='utf-8') as fh:
        nbformat.write(nb_in, fh)

    return nb_in, nb_out

if __name__ == "__main__":

    from run_notebook import run_notebook
    basepath = '/orange/adamginsburg/jwst/brick/'

    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/BrA_Separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/F466_separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_nrcb.ipynb')

    run_notebook(f'{basepath}/notebooks/PaA_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/PaA_Separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/F212_Separation_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/F212_Separation_nrcb.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrca.ipynb')
    run_notebook(f'{basepath}/notebooks/StarDestroyer_PaA_nrcb.ipynb')

    run_notebook(f'{basepath}/notebooks/Stitch_A_to_B.ipynb')
