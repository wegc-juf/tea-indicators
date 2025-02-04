import tkinter as tk
from tkinter import ttk
import os


def show_parameters(opts):
    # Create a new window
    window = tk.Tk()
    window.title('CFG Parameters')

    # Create a frame for the Treeview widget
    frame = tk.Frame(window)
    frame.pack(pady=10)

    num_of_entries = len(vars(opts))
    tree_height = num_of_entries if num_of_entries > 0 else 1

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

    # Define the column headings
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
                                                            '../TEA_CFG.yaml'))
    edit_button.pack(side='left', padx=10, pady=10)

    # Run the GUI
    window.mainloop()


def edit_parameters(window, opts, yaml_fname):
    """
        Open a new window to edit parameters.
        Args:
            window: The main window to close when finished.
            opts: Namespace object to edit.
            yaml_fname: Name of the YAML file to update after editing.
        """
    edit_window = tk.Toplevel()  # Create a new window on top of the main window
    edit_window.title('Edit Parameters')

    # Store the variables linked to the entry fields
    entries = {}

    # Create and populate the form with the current parameters and values
    for index, (name, value) in enumerate(vars(opts).items()):
        tk.Label(edit_window, text=name).grid(row=index, column=0)
        entry_var = tk.StringVar(value=value)  # Create a variable for the entry
        tk.Entry(edit_window, textvariable=entry_var).grid(row=index, column=1)
        entries[name] = entry_var  # Store it to retrieve the data later

    # Confirm Button in the edit window
    def confirm_edit():
        # Update the opts namespace with new values
        for name, entry in entries.items():
            new_value = entry.get()
            try:
                # Try to infer the correct type by evaluating the value
                new_value = eval(new_value)
            except (NameError, SyntaxError):
                pass  # Keep the value as string if it cannot be evaluated
            setattr(opts, name, new_value)
        update_yaml(yaml_fname, opts)  # Update the YAML file
        edit_window.destroy()  # Close the edit window
        window.destroy()  # Close the main window
        show_parameters(opts)  # Optionally re-open the main window to show updated values

    tk.Button(edit_window, text='Confirm',
              command=confirm_edit).grid(row=len(entries), column=1)


def update_yaml(fname, opts):
    """
    Update the YAML file with new parameters.
    Args:
        fname: File name of the YAML config.
        opts: Namespace containing the updated parameters.
    """
    new_params = vars(opts)

    new_name = '../NEW_' + fname.split('/')[1]

    with open(fname, 'r') as original_file, open(new_name, 'w') as new_file:
        # Iterate through each line in the original file
        for line in original_file:
            if line == '\n':
                new_file.write(line)
                continue
            key = line.split(':')[0].strip()
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