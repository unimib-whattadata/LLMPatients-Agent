from pymongo import MongoClient
import os, json

class PatientDB:
    def __init__(self, mongo_uri="mongodb://localhost:27017/", 
                 db_name="patient_db", 
                 collection_name="patients", 
                 patients_dir="data/patients"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.patients_dir = patients_dir

        # Ensure indexes for fast search
        self.collection.create_index("id", unique=True)
        self.collection.create_index("clinical_profile.primary_diagnoses")
        self.collection.create_index("clinical_profile.comorbid_features")

    def load_all_patients(self):
        """Load or update all patient JSONs from the directory into MongoDB."""
        for filename in os.listdir(self.patients_dir):
            if filename.endswith(".json"):
                filepath = os.path.join(self.patients_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    patient = json.load(f)
                self.upsert_patient(patient)

    def upsert_patient(self, patient):
        """Insert or update a single patient record."""
        self.collection.update_one(
            {"id": patient["id"]},
            {"$set": patient},
            upsert=True
        )

    def get_patient_by_id(self, patient_id):
        """Retrieve a patient by ID."""
        return self.collection.find_one({"id": patient_id})

    def find_patients_by_condition(self, condition):
        """Find patients by a given condition (diagnosis or comorbid feature)."""
        results = self.collection.find({
            "$or": [
                {"clinical_profile.primary_diagnoses": condition},
                {"clinical_profile.comorbid_features": condition}
            ]
        })
        return [(p["id"], p["demographics"]["name"]) for p in results]

    def list_all_patients(self):
        """Return IDs and names of all patients."""
        return [(p["id"], p["demographics"]["name"]) for p in self.collection.find()]

    def delete_patient(self, patient_id):
        """Remove a patient by ID."""
        self.collection.delete_one({"id": patient_id})