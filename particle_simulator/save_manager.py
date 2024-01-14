import os
import pickle
from tkinter.filedialog import asksaveasfilename, askopenfilename
from typing import Optional

from . import sim_pickle
from .controller_state import ControllerState


class SaveManager:
    def __init__(self, file_location: Optional[str] = None):
        self.file_location = file_location
        self.filename = "simulation"

    def save(
        self, controller_state: ControllerState, filename: Optional[str] = None
    ) -> None:
        if filename is None:
            filename = asksaveasfilename(
                initialdir=self.file_location,
                initialfile=self.filename,
                defaultextension=".sim",
                filetypes=[("Simulation files", "*.sim"), ("All Files", "*.*")],
            )

            if not filename:
                return
        data = sim_pickle.to_dict(controller_state)
        with open(filename, "wb") as file_object:
            pickle.dump(data, file_object)

        self.file_location, self.filename = os.path.split(filename)

    def load(self, filename: Optional[str] = None) -> Optional[ControllerState]:
        if filename is None:
            filename = askopenfilename(
                initialdir=self.file_location,
                initialfile=self.filename,
                defaultextension=".sim",
                filetypes=[("Simulation files", "*.sim"), ("All Files", "*.*")],
            )

            if not filename:
                return None

        with open(filename, "rb") as file_object:
            data = pickle.load(file_object)

        self.file_location, self.filename = os.path.split(filename)
        return sim_pickle.from_dict(data)
