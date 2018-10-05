from Flask import flask, requests
import json
import sqlite3
import http.client, urllib.request, urllib.parse, urllib.error, base64


db_old = sqlite3.connect("old.db")  # database for old users
cursor_old = db_old.cursor()


@app.route('/', methods=['POST'])
def receive_old():
    global faceId1  # contains url for picture sent by app captured for verification
    faceId1 = request.json['verification_pic']  # we need to assign userID to everyone as names may
# be common with other users
    command_old_people = ('''SELECT profile_pic
                             FROM Old;''')  # sql command
    row = cursor_old.execute(command_old_people)
    faceId2 = row[0]
    headers = {
        # Request headers
        'Content-Type': 'application/json',
        'Ocp-Apim-Subscription-Key': '{subscription key}',
    }
    params = urllib.parse.urlencode({
        'faceId1': faceId1,
        'faceId2': faceId2,
    })
    return face_matcher(params, headers)


def face_matcher(params, headers):
    try:
        conn = http.client.HTTPSConnection('westus.api.cognitive.microsoft.com')
        conn.request("POST", "/face/v1.0/verify?%s" % params, "{body}", headers)
        response = conn.getresponse()
        data = response.read()
        conn.close()
        return data
    except Exception as e:
        print("[Errno {0}] {1}".format(e.errno, e.strerror))