import os
import pickle
from tkinter.filedialog import asksaveasfilename, askopenfilename
from typing import Optional

from .error import Error


class SaveManager:
    def __init__(self, sim):
        self.sim = sim
        self.file_location = os.path.dirname(self.sim.gui.path)
        self.filename = "simulation"

    def save(self, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = asksaveasfilename(
                initialdir=self.file_location,
                initialfile=self.filename,
                defaultextension=".sim",
                filetypes=[("Simulation files", "*.sim"), ("All Files", "*.*")],
            )

            if not filename:
                return
        try:
            data = self.sim.to_dict()
            with open(filename, "wb") as file_object:
                pickle.dump(data, file_object)

            self.file_location, self.filename = os.path.split(filename)
        except Exception as error:
            self.sim.error = Error("Saving-Error", error)

    def load(self, filename: Optional[str] = None) -> None:
        if not self.sim.paused:
            self.sim.toggle_paused()
        if filename is None:
            filename = askopenfilename(
                initialdir=self.file_location,
                initialfile=self.filename,
                defaultextension=".sim",
                filetypes=[("Simulation files", "*.sim"), ("All Files", "*.*")],
            )

            if not filename:
                return
        try:
            with open(filename, "rb") as file_object:
                data = pickle.load(file_object)

            self.sim.from_dict(data)
            self.file_location, self.filename = os.path.split(filename)
        except Exception as error:
            self.sim.error = Error("Loading-Error", error)
