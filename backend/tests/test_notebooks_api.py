"""
Tests for the notebooks endpoints (CRUD, ownership, isolation).
"""
import uuid


# ---------------------------------------------------------------- helpers

def make_notebook(client, headers, name=None, **overrides):
    payload = {"name": name or f"NB {uuid.uuid4().hex[:6]}", **overrides}
    r = client.post("/api/notebooks", json=payload, headers=headers)
    assert r.status_code == 201, r.text
    return r.json()


# ---------------------------------------------------------------- create

def test_create_notebook_returns_full_shape(client, user_a):
    nb = make_notebook(client, user_a["headers"], name="AI Course", description="d", color="mint")
    assert nb["name"] == "AI Course"
    assert nb["description"] == "d"
    assert nb["color"] == "mint"
    assert nb["is_pinned"] is False
    assert nb["source_count"] == 0
    assert nb["question_count"] == 0
    assert "id" in nb and "created_at" in nb and "updated_at" in nb


def test_create_notebook_requires_auth(client):
    r = client.post("/api/notebooks", json={"name": "Nope"})
    assert r.status_code == 401


def test_create_notebook_rejects_blank_name(client, user_a):
    r = client.post("/api/notebooks", json={"name": "   "}, headers=user_a["headers"])
    assert r.status_code == 422


def test_create_notebook_rejects_long_name(client, user_a):
    r = client.post("/api/notebooks", json={"name": "x" * 121}, headers=user_a["headers"])
    assert r.status_code == 422


# ---------------------------------------------------------------- read

def test_list_notebooks_returns_own_notebooks(client, user_a, user_b):
    mine = make_notebook(client, user_a["headers"], name="Mine Only")
    theirs = make_notebook(client, user_b["headers"], name="Theirs Only")

    r = client.get("/api/notebooks", headers=user_a["headers"])
    assert r.status_code == 200
    ids = [n["id"] for n in r.json()["notebooks"]]
    assert mine["id"] in ids
    assert theirs["id"] not in ids  # isolation between users


def test_get_notebook_detail_includes_collections(client, user_a):
    nb = make_notebook(client, user_a["headers"])
    r = client.get(f"/api/notebooks/{nb['id']}", headers=user_a["headers"])
    assert r.status_code == 200
    body = r.json()
    assert body["documents"] == []
    assert body["notes"] == []
    assert body["saved_answers"] == []


def test_get_missing_notebook_404(client, user_a):
    r = client.get("/api/notebooks/99999999", headers=user_a["headers"])
    assert r.status_code == 404


def test_cannot_read_another_users_notebook(client, user_a, user_b):
    theirs = make_notebook(client, user_b["headers"])
    r = client.get(f"/api/notebooks/{theirs['id']}", headers=user_a["headers"])
    assert r.status_code == 404  # ownership filter hides it, not 403


# ---------------------------------------------------------------- update

def test_update_notebook_fields(client, user_a):
    nb = make_notebook(client, user_a["headers"])
    r = client.patch(
        f"/api/notebooks/{nb['id']}",
        json={"name": "Renamed", "color": "coral", "is_pinned": True},
        headers=user_a["headers"],
    )
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "Renamed"
    assert body["color"] == "coral"
    assert body["is_pinned"] is True


def test_cannot_update_another_users_notebook(client, user_a, user_b):
    theirs = make_notebook(client, user_b["headers"])
    r = client.patch(
        f"/api/notebooks/{theirs['id']}",
        json={"name": "Hacked"},
        headers=user_a["headers"],
    )
    assert r.status_code == 404
    # Confirm unchanged
    r = client.get(f"/api/notebooks/{theirs['id']}", headers=user_b["headers"])
    assert r.json()["name"] != "Hacked"


# ---------------------------------------------------------------- delete

def test_delete_notebook(client, user_a):
    nb = make_notebook(client, user_a["headers"])
    r = client.delete(f"/api/notebooks/{nb['id']}", headers=user_a["headers"])
    assert r.status_code == 204
    r = client.get(f"/api/notebooks/{nb['id']}", headers=user_a["headers"])
    assert r.status_code == 404


def test_cannot_delete_another_users_notebook(client, user_a, user_b):
    theirs = make_notebook(client, user_b["headers"])
    r = client.delete(f"/api/notebooks/{theirs['id']}", headers=user_a["headers"])
    assert r.status_code == 404
    # Still exists for the owner
    r = client.get(f"/api/notebooks/{theirs['id']}", headers=user_b["headers"])
    assert r.status_code == 200
