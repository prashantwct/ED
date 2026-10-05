"""Invented damage surveys in the shape of the real Epicollect5 projects.

The question keys match the live herd-crop-damage and herd-house-damage
forms. Every person, number and remark here is made up: the real
exports carry owners' names and phone numbers and never belong in the
repository.
"""

import csv
import io
import zipfile

# Inside the Kaseru beat (Pataur Core, Bandhavgarh NP); and a point far
# outside every boundary layer.
INSIDE = {"latitude": 23.81739, "longitude": 80.98181, "accuracy": 4}
NOWHERE = {"latitude": 21.0, "longitude": 76.0, "accuracy": 4}


def crop_entry(i, uploaded="2026-09-01T00:00:00.000Z", location=INSIDE, *,
               owner=None, mobile=None, damage="2026-09-10", survey="2026-09-14",
               status="Not Applied", value=5000, elephants=3, remark="",
               crops="Rice", fencing="Bamboo fence", stage="Flowering Stage"):
    return {
        "ec5_uuid": f"crop-{i}",
        "created_at": "2026-09-14T08:00:00.000Z",
        "uploaded_at": uploaded,
        "created_by": "surveyor@example.org",
        "title": "Testgaon",
        "1_Location": dict(location),
        "2_Survey_Date": f"{survey}T00:00:00.000",
        "3_Survey_Time": "1970-01-01T13:00:00.000",
        "4_Village": "Testgaon",
        "5_Name_of_the_Owner": owner or f"Owner {i}",
        "6_Mobile_Number": mobile or f"90000000{i:02d}",
        "7_Date_of_Damage": f"{damage}T00:00:00.000",
        "8_Time_of_Damage": "1970-01-01T22:30:00.000",
        "10_Total_Land": 2,
        "11_Land_Cultivated": 1.5,
        "12_Irrigation": "Yes",
        "13_Type_of_Farming": "Single Crop",
        "14_Type_of_Fencing": fencing,
        "15_Crops": crops,
        "16_Stage_of_Crop": stage,
        "17_Damage_Area_Owner": 0.4,
        "18_Damage_Area_Colle": 0.3,
        "19_Damage_Area_Calcu": 0.2,
        "20_Quantity_of_crop_": 2,
        "21_Estimated_value_o": value,
        "22_Compensation_rece": status,
        "23_Compensation_amou": "",
        "24_Satisfaction_with": "",
        "25_Number_of_elephan": elephants,
        "26_Distance_to_neare": 200,
        "27_Distance_to_neare": 100,
        "28_Photo": f"crop-{i}.jpg",
        "29_Remarks": remark,
    }


def house_entry(i, uploaded="2026-08-01T00:00:00.000Z", location=INSIDE, *,
                damage="2026-07-09", survey="2026-07-10", status="In process",
                value=10000, elephants=1, light="No", rooms="Kitchen, Hall",
                house_type="Semi- Pucca", remark=""):
    return {
        "ec5_uuid": f"house-{i}",
        "created_at": "2026-07-10T08:00:00.000Z",
        "uploaded_at": uploaded,
        "created_by": "surveyor@example.org",
        "title": "Testgaon",
        "1_Location": dict(location),
        "2_Survey_Date": f"{survey}T00:00:00.000",
        "3_Survey_Time": "1970-01-01T13:00:00.000",
        "4_Village": "testgaon",
        "5_Name_of_the_Owner": f"House Owner {i}",
        "6_Mobile_Number": f"80000000{i:02d}",
        "7_Date_of_Damage": f"{damage}T00:00:00.000",
        "8_Time_of_Damage": "1970-01-01T01:00:00.000",
        "10_Type_of_House": house_type,
        "11_Was_the_light_on_": light,
        "12_Total_Area_of_Hou": 500,
        "13_Damaged_area_of_h": 100,
        "14_Which_area_of_hou": rooms,
        "15_Presence_of_store": "Yes",
        "16_Presence_of_store": "No",
        "17_Estimated_cost_of": value,
        "18_Compensation_rece": status,
        "19_Compensation_amou": "",
        "20_Satisfaction_with": "",
        "21_Number_of_elephan": elephants,
        "22_Distance_to_neare": 100,
        "23_Distance_to_neare": 50,
        "24_Photo": f"house-{i}.jpg",
        "25_Remarks": remark,
    }


def export_csv(entries):
    """Entries as the project page's CSV export writes them.

    Location splits into lat_/long_/accuracy_ columns, dates become
    dd/mm/yyyy and times HH:MM, and the file starts with a byte-order mark.
    """
    rows = []
    for entry in entries:
        row = {}
        for key, value in entry.items():
            if isinstance(value, dict):
                row[f"lat_{key}"] = value["latitude"]
                row[f"long_{key}"] = value["longitude"]
                row[f"accuracy_{key}"] = value["accuracy"]
            elif isinstance(value, str) and value.startswith("1970-01-01T"):
                row[key] = value[11:16]
            elif isinstance(value, str) and len(value) > 10 and value[4] == "-" \
                    and value.endswith("T00:00:00.000"):
                y, m, d = value[:10].split("-")
                row[key] = f"{d}/{m}/{y}"
            else:
                row[key] = value
        rows.append(row)
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return ("﻿" + out.getvalue()).encode("utf-8")


def export_zip(name, entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(f"form-1__{name}.csv", export_csv(entries))
    return buffer.getvalue()
