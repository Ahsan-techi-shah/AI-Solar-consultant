957
958
959
960
961
962
963
964
965
966
967
968
969
970
971
972
973
974
975
976
977
978
979
980
981
982
983
984
985
986
987
988
989
990
991
992
993
994
995
996
997
998
999
1000
1001
1002
1003
1004
1005
1006
1007
1008
1009
1010
1011
1012
1013
import base64


# =========================================================
# QUOTATION
# =========================================================

def build_quote(
    result,
    rate,
    inverter_price,
    installation,
):
    kw = result["sizing"]["recommended_kw"]
    panel_w = result["selection"]["panel_watt"]
    count = result["sizing"]["panel_count"]

    panel_cost = kw * 1000 * rate

    total = (
        panel_cost
        + inverter_price
        + installation
    )

    items = [
        {
            "Item": result["selection"]["panel_brand"],
            "Specification": f"{panel_w} W",
            "Quantity": count,
            "Amount PKR": round(panel_cost),
        },
        {
            "Item": result["selection"]["inverter_brand"],
            "Specification": "Approved inverter",
            "Quantity": 1,
            "Amount PKR": round(inverter_price),
        },
        {
            "Item": "Installation / BOS / other",
            "Specification": "",
            "Quantity": 1,
            "Amount PKR": round(installation),
        },
    ]

    text = (
        f'Customer: {result["customer"]["name"]}\n'
        f"Recommended system: {kw:.1f} kW\n"
        f"Total: PKR {total:,.0f}\n"
    )

    return {
        "items": items,
        "total": total,
        "text": text,
    }
