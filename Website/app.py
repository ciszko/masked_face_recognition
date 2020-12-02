from flask import Flask, render_template, flash, redirect, request, url_for
from werkzeug.utils import secure_filename
import os
import datetime
import json
import random

app = Flask(__name__)
app.secret_key = "SECRET"

UPLOAD_FOLDER = "./upload_folder"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp"}

app.config.update(TEMPLATES_AUTO_RELOAD=True, UPLOAD_FOLDER=UPLOAD_FOLDER, DEBUG=True)


@app.route("/")
def index():
    lang = request.args.get("lang")
    if lang is None:
        lang = "pl"
    return render_template("index.html", lang=lang)


@app.route("/thank_you")
def thankyou():
    lang = request.args.get("lang")
    if lang is None:
        lang = "pl"
    thanks = choose_thanks()
    return render_template("thank_you.html", data=thanks, lang=lang)


@app.route("/klauzula")
def klauzula():
    lang = request.args.get("lang")
    if lang is None:
        lang = "pl"
    return render_template("klauzula.html", lang=lang)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def choose_thanks():
    with open("./thankyou.json", "r") as f:
        data = json.load(f)
    return data["thanks"][random.randrange(0, len(data["thanks"]))]


@app.route("/form/", methods=["GET", "POST"])
def form():
    lang = request.args.get("lang")
    if lang is None:
        lang = "pl"
    if request.method == "POST":
        if "file[]" not in request.files:
            flash("Brak zdjęcia w żądaniu", "danger")
            return redirect(request.url)
        file_list = request.files.getlist("file[]")
        if len(file_list) < 2:
            flash("Minimalna ilość zdjęć to 2", "danger")
            return redirect(request.url)
        print(file_list)
        for file in file_list:
            print(file)
            if file.filename == "":
                flash("Nie wybrano zdjęcia", "danger")
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                to_save = datetime.datetime.now().strftime("%m_%d_%H_%M") + filename
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], to_save))
            else:
                flash("Dozwolone formaty plików to: png, jpg, jpeg, bmp", "danger")
                return redirect(request.url)
        thanks = choose_thanks()
        return render_template("thank_you.html", data=thanks, lang=lang)

    return render_template("form.html", lang=lang)


if __name__ == "__main__":
    app.run("0.0.0.0")
