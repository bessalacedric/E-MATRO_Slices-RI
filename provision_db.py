"""
provision_db.py
===============
Insert all 12 UE subscriptions into the OAI MySQL DB.
Run once after each new SLICES-RI reservation, after loading oai_db.sql.

Handles the case where oai_db.sql only seeds AuthenticationSubscription —
inserts UE1 seed rows into the other 3 tables if missing, then copies
all rows for UE2-12.
"""

import subprocess

IMSIS = [
    "001010000000101", "001010000000102", "001010000000103",
    "001010000000104", "001010000000105", "001010000000106",
    "001010000000107", "001010000000108", "001010000000109",
    "001010000000110", "001010000000111", "001010000000112",
]
SOURCE = "001010000000101"

# Seed data for UE1 in tables that oai_db.sql doesn't populate
UE1_SEEDS = {
    "AccessAndMobilitySubscriptionData": {
        "cols":   "(ueid, servingPlmnid, nssai)",
        "values": (
            f"('{SOURCE}', '00101', "
            "'{\"defaultSingleNssais\":[{\"sst\":1,\"sd\":\"16777215\"}],"
            "\"singleNssais\":[{\"sst\":1,\"sd\":\"1\"}]}')"
        ),
    },
    "SessionManagementSubscriptionData": {
        "cols": "(ueid, servingPlmnid, singleNssai, dnnConfigurations)",
        "values": (
            f"('{SOURCE}', '00101', "
            "'{\"sst\":1,\"sd\":\"16777215\"}', "
            "'{\"oai\":{\"pduSessionTypes\":{\"defaultSessionType\":\"IPV4\"},"
            "\"sscModes\":{\"defaultSscMode\":\"SSC_MODE_1\"},"
            "\"5gQosProfile\":{\"5qi\":9,\"arp\":{\"priorityLevel\":8,"
            "\"preemptCap\":\"NOT_PREEMPT\",\"preemptVuln\":\"NOT_PREEMPTABLE\"},"
            "\"priorityLevel\":8},\"sessionAmbr\":{\"uplink\":\"10Gbps\","
            "\"downlink\":\"10Gbps\"},\"staticIpAddress\":[]}}')"
        ),
    },
    "SmfSelectionSubscriptionData": {
        "cols":   "(ueid, servingPlmnid, subscribedSnssaiInfos)",
        "values": (
            f"('{SOURCE}', '00101', "
            "'[{\"sst\":1,\"sd\":\"16777215\"},{\"sst\":1,\"sd\":\"1\"}]')"
        ),
    },
}

TABLES = [
    "AuthenticationSubscription",
    "AccessAndMobilitySubscriptionData",
    "SessionManagementSubscriptionData",
    "SmfSelectionSubscriptionData",
]


def run_sql(sql: str) -> bool:
    r = subprocess.run(
        ["docker", "exec", "mysql", "mysql",
         "-u", "root", "--password=linux", "oai_db", "-e", sql],
        capture_output=True, text=True
    )
    return r.returncode == 0


def query(sql: str) -> str:
    return subprocess.getoutput(
        f"docker exec mysql mysql -u root --password=linux oai_db "
        f"-sNe \"{sql}\" 2>/dev/null"
    ).strip()


def get_columns(table: str) -> list:
    out = query(
        f"SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        f"WHERE TABLE_SCHEMA='oai_db' AND TABLE_NAME='{table}' "
        f"ORDER BY ORDINAL_POSITION;"
    )
    return [c for c in out.split('\n') if c.strip()]


def main():
    print("=== E-MATRO DB Provisioning ===\n")

    # Step 1: ensure UE1 exists in all 4 tables
    print("Step 1 — Seeding UE1 in all tables:")
    for tbl, seed in UE1_SEEDS.items():
        count = query(f"SELECT COUNT(*) FROM {tbl} WHERE ueid='{SOURCE}';")
        if count == '0':
            sql = (f"INSERT IGNORE INTO {tbl} {seed['cols']} "
                   f"VALUES {seed['values']};")
            ok = run_sql(sql)
            print(f"  {tbl}: seeded ✓" if ok else f"  {tbl}: FAILED")
        else:
            print(f"  {tbl}: UE1 already present ({count} row)")

    print()

    # Step 2: copy UE1 rows for UE2-12 in all 4 tables
    print("Step 2 — Copying to UE2-12:")
    for tbl in TABLES:
        cols = get_columns(tbl)
        if not cols:
            print(f"  {tbl}: could not read columns — skipping")
            continue

        count = query(f"SELECT COUNT(*) FROM {tbl} WHERE ueid='{SOURCE}';")
        if count == '0':
            print(f"  {tbl}: no UE1 row — skipping")
            continue

        ok_count = 0
        for imsi in IMSIS[1:]:   # skip UE1
            parts = []
            for c in cols:
                if c in ('ueid', 'supi'):
                    parts.append(f"'{imsi}'")
                else:
                    parts.append(c)
            sql = (f"INSERT IGNORE INTO {tbl} ({','.join(cols)}) "
                   f"SELECT {','.join(parts)} "
                   f"FROM {tbl} WHERE ueid='{SOURCE}';")
            if run_sql(sql):
                ok_count += 1

        total = query(f"SELECT COUNT(*) FROM {tbl};")
        print(f"  {tbl}: {total} rows total ✓")

    print()
    print("Step 3 — Final verification:")
    for tbl in TABLES:
        n = query(f"SELECT COUNT(*) FROM {tbl};")
        status = "✓" if int(n or 0) >= 12 else "⚠ only " + n
        print(f"  {tbl}: {n} {status}")

    print("\nDB provisioning complete ✓")


if __name__ == "__main__":
    main()

