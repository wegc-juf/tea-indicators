import argparse
import tkinter as tk
from tkinter import ttk
import os
import yaml


def _getopts():
    """
    get command line arguments

    Returns:
        opts: command line parameters
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--config-file', '-cf',
                        dest='config_file',
                        type=str,
                        default='../TEA_CFG.yaml',
                        help='TEA configuration file (default: TEA_CFG.yaml)')

    myopts = parser.parse_args()

    return myopts


def run_gui(opts):
    """
    Create a window that lists all currently defined CFG parameter
    Args:
        opts: CFG parameter

    Returns:

    """

    # Create a new window
    window = tk.Tk()
    window.title('CFG Parameters')

    # Create a frame for the Treeview widget
    frame = tk.Frame(window)
    frame.pack(pady=10)

    # Find height for tree
    num_of_entries = len(vars(opts))
    tree_height = num_of_entries if num_of_entries > 0 else 1

    # Find column widths
    vcol_width = 0
    for ivar in vars(opts):
        if len(ivar) > vcol_width:
            vcol_width = len(ivar) * 75

    pcol_width = 0
    for ivar in vars(opts).keys():
        if len(ivar) > pcol_width:
            pcol_width = len(ivar) * 30

    # Create a treeview to display the parameters and their values
    tree = ttk.Treeview(frame, columns=('Parameter', 'Value'), show='headings',
                        height=tree_height)

    # Define the column headings and the column widths
    tree.heading('Parameter', text='Parameter')
    tree.heading('Value', text='Value')
    tree.column('Parameter', width=pcol_width)
    tree.column('Value', width=vcol_width)

    # Adding each parameter and its value to the tree as a new row
    for name, value in vars(opts).items():
        tree.insert('', tk.END, values=(name, value))

    # Pack the treeview into the frame
    tree.pack()

    # Add the buttons at the bottom of the same window
    button_frame = tk.Frame(window)
    button_frame.pack(fill='x', expand=True)

    # Confirm Button
    confirm_button = tk.Button(button_frame, text='Confirm', command=window.destroy)
    confirm_button.pack(side='right', padx=10, pady=10)

    # Edit Button
    edit_button = tk.Button(button_frame, text='Edit parameter',
                            command=lambda: edit_parameters(window,
                                                            opts,
                                                            opts.cfg_file))
    edit_button.pack(side='left', padx=10, pady=10)

    # Run the GUI
    window.mainloop()


def edit_parameters(window, opts, yaml_fname):
    """
    Opens a new window to edit CFG parameter
    Args:
        window: old window showing the CFG parameter
        opts: CFG parameter
        yaml_fname: Name of yaml (CFG) file

    Returns:

    """
    # Create a new window on top of the main window
    edit_window = tk.Toplevel()
    edit_window.title('Edit Parameters')

    # Store the variables linked to the entry fields
    entries = {}

    # Create and populate the form with the current parameters and values
    for index, (name, value) in enumerate(vars(opts).items()):
        tk.Label(edit_window, text=name).grid(row=index, column=0)

        # Convert boolean values to string for display
        if isinstance(value, bool) or ',' in str(value):
            value = str(value)

        # Create a variable for the entry
        entry_var = tk.StringVar(value=value)
        tk.Entry(edit_window, textvariable=entry_var, width=75).grid(row=index, column=1)
        # Store it to retrieve the data later
        entries[name] = entry_var

    # Confirm Button in the edit window
    def confirm_edit():
        # Update the opts namespace with new values
        for my_name, entry in entries.items():
            new_value = entry.get().strip()
            if '-' in new_value and new_value.count('-') == 1:
                parts = new_value.split('-')
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    # If both parts are digits, keep it as a string
                    new_value = new_value
                else:
                    # Handle as a normal string if not valid
                    new_value = str(new_value)
            elif ',' in new_value and new_value.count(',') == 2:
                parts = new_value.split(',')
                if len(parts) == 3 and all(part.isdigit() for part in parts):
                    # If all parts are digits, keep it as a string
                    new_value = new_value
                else:
                    # Handle as a normal string if not valid
                    new_value = str(new_value)
            else:
                try:
                    # Try to infer the correct type by evaluating the value
                    new_value = eval(new_value)
                except (NameError, SyntaxError):
                    # Keep the value as string if it cannot be evaluated
                    pass
            setattr(opts, my_name, new_value)
        # Update the YAML file
        update_yaml(yaml_fname, opts)
        # Close all windows and open main window again
        edit_window.destroy()
        window.destroy()
        run_gui(opts)

    tk.Button(edit_window, text='Confirm',
              command=confirm_edit).grid(row=len(entries), column=1)


def update_yaml(fname, opts):
    """
    Update the YAML file with new parameters.
    Args:
        fname: File name of the YAML config.
        opts: Namespace containing the updated parameters.
    """

    # Create dict with newly set CFG parameters
    new_params = vars(opts)

    # fix threshold_type if 'abs' was selected (somehow gets converted to builtin_function_or_method)
    if new_params['threshold_type'] not in ['perc', 'abs']:
        new_params['threshold_type'] = 'abs'

    # Create a filename for the new CFG file
    new_name = '../NEW_' + fname.split('/')[1]

    scripts = ['regrid_SPARTACUS_to_WEGNext', 'create_region_masks',
               'create_static_files', 'calc_TEA', 'calc_station_TEA',
               'calc_amplification_factors', 'calc_AGR_vars']

    # Open old CFG file and check for new values
    with open(fname, 'r') as original_file, open(new_name, 'w') as new_file:
        # Iterate through each line in the original file
        for line in original_file:
            if line == '\n':
                new_file.write(line)
                continue
            key = line.split(':')[0].strip()
            if key[0] == '#':
                new_file.write(line)
                continue
            if key in scripts:
                sec = key
            ovalue = line.split(':')[1].strip()

            if key not in new_params.keys():
                new_file.write(line)
                continue
            nvalue = new_params[key]
            if ovalue != nvalue:
                if ovalue == 'null' and nvalue == '':
                    new_file.write(line)
                    continue
                if ovalue[0] == '&':
                    new_file.write(line)
                    continue
                if ovalue == 'false' and nvalue == 0:
                    new_file.write(line)
                    continue
                if ovalue == 'true' and nvalue == 1:
                    new_file.write(line)
                    continue
                modified_line = line.replace(ovalue, f'{nvalue}')
                new_file.write(modified_line)
            else:
                new_file.write(line)

    os.system(f'mv {new_name} {fname}')


def flatten_yaml(data, parent_key='', sep='_'):
    items = {}
    for key, value in data.items():
        new_key = f"{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_yaml(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def run():
    # get command line parameters
    cmd_opts = _getopts()

    # load CFG parameters
    with open(cmd_opts.config_file, 'r') as stream:
        opts = yaml.safe_load(stream)

    # Convert the loaded YAML data to a Namespace object
    flattened_data = flatten_yaml(opts)
    # Create a dictionary with all parameters
    params = {key: value for key, value in flattened_data.items() if value is not None}
    opts = argparse.Namespace(**params)

    # add name of CFG file
    opts.cfg_file = cmd_opts.config_file

    # run gui to show parameters
    run_gui(opts)


if __name__ == '__main__':
    run()
