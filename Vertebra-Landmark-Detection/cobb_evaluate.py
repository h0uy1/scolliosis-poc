###########################################################################################
## This code is transfered from matlab version of the MICCAI challenge
## Oct 1 2019
###########################################################################################
import numpy as np
import cv2


def is_S(mid_p_v):
    # mid_p_v:  34 x 2
    ll = []
    num = mid_p_v.shape[0]
    for i in range(num-2):
        term1 = (mid_p_v[i, 1]-mid_p_v[num-1, 1])/(mid_p_v[0, 1]-mid_p_v[num-1, 1])
        term2 = (mid_p_v[i, 0]-mid_p_v[num-1, 0])/(mid_p_v[0, 0]-mid_p_v[num-1, 0])
        ll.append(term1-term2)
    ll = np.asarray(ll, np.float32)[:, np.newaxis]   # 32 x 1
    ll_pair = np.matmul(ll, np.transpose(ll))        # 32 x 32
    a = sum(sum(ll_pair))
    b = sum(sum(abs(ll_pair)))
    if abs(a-b)<1e-4:
        return False
    else:
        return True


def _clip_extended_line(p1, p2, w, h, extend=2000):
    vec = p2 - p1
    norm = np.sqrt(np.sum(vec ** 2))
    if norm < 1e-6:
        pt1 = (int(p1[0]), int(p1[1]))
        return pt1, pt1

    direction = vec / norm
    line_start = p1 - direction * extend
    line_end = p2 + direction * extend
    ok, clipped_start, clipped_end = cv2.clipLine((0, 0, w, h),
                                                  (int(line_start[0]), int(line_start[1])),
                                                  (int(line_end[0]), int(line_end[1])))
    if ok:
        return clipped_start, clipped_end

    return (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))


def _put_label(image, text, org, color, font_scale=0.7, thickness=2):
    h, w = image.shape[:2]
    x = max(8, min(int(org[0]), w - 190))
    y = max(24, min(int(org[1]), h - 12))
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)


def _cobb_geometry(pts, image_shape):
    h, w = image_shape[:2]
    pts = np.asarray(pts, np.float32)
    num_pts = pts.shape[0]
    vnum = num_pts // 4 - 1

    mid_p_v = (pts[0::2, :] + pts[1::2, :]) / 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i, :] + pts[i + 2, :]) / 2
        pt2 = (pts[i + 1, :] + pts[i + 3, :]) / 2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)

    vec_m = mid_p[1::2, :] - mid_p[0::2, :]
    dot_v = np.matmul(vec_m, np.transpose(vec_m))
    mod_v = np.sqrt(np.sum(vec_m ** 2, axis=1))[:, np.newaxis]
    mod_v = np.matmul(mod_v, np.transpose(mod_v))
    cosine_angles = np.clip(dot_v / mod_v, a_min=0., a_max=1.)
    angles = np.arccos(cosine_angles)
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)

    cobb_angle1 = np.amax(maxt) / np.pi * 180
    main_pair = (pos2, pos1[pos2])
    angle_pairs = [('Main', cobb_angle1, main_pair[0], main_pair[1])]

    flag_s = is_S(mid_p_v)
    if not flag_s:
        cobb_angle2 = angles[0, pos2] / np.pi * 180
        cobb_angle3 = angles[vnum, pos1[pos2]] / np.pi * 180
        angle_pairs.append(('Upper', cobb_angle2, 0, pos2))
        angle_pairs.append(('Lower', cobb_angle3, vnum, pos1[pos2]))
    else:
        if (mid_p_v[pos2 * 2, 1] + mid_p_v[pos1[pos2] * 2, 1]) < h:
            angle2 = angles[pos2, :(pos2 + 1)]
            cobb_angle2 = np.max(angle2) / np.pi * 180
            pos1_1 = np.argmax(angle2)

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum + 1)]
            cobb_angle3 = np.max(angle3) / np.pi * 180
            pos1_2 = np.argmax(angle3) + pos1[pos2] - 1
            angle_pairs.append(('Upper', cobb_angle2, pos2, pos1_1))
            angle_pairs.append(('Lower', cobb_angle3, pos1[pos2], pos1_2))
        else:
            angle2 = angles[pos2, :(pos2 + 1)]
            cobb_angle2 = np.max(angle2) / np.pi * 180
            pos1_1 = np.argmax(angle2)

            angle3 = angles[pos1_1, :(pos1_1 + 1)]
            cobb_angle3 = np.max(angle3) / np.pi * 180
            pos1_2 = np.argmax(angle3)
            angle_pairs.append(('Upper', cobb_angle2, pos2, pos1_1))
            angle_pairs.append(('Lower', cobb_angle3, pos1_1, pos1_2))

    return mid_p, angle_pairs


def draw_cobb_overlay(pts, image):
    pts = np.asarray(pts, np.float32)
    h, w = image.shape[:2]
    mid_p, angle_pairs = _cobb_geometry(pts, image.shape)

    vertebra_pts = pts.reshape((-1, 4, 2))
    centers = np.mean(vertebra_pts, axis=1).astype(np.int32)

    for vertebra in vertebra_pts:
        points = vertebra.astype(np.int32)
        cv2.polylines(image, [points[[0, 1, 3, 2]].reshape((-1, 1, 2))],
                      isClosed=True, color=(255, 180, 40), thickness=1, lineType=cv2.LINE_AA)
        for point in points:
            cv2.circle(image, (int(point[0]), int(point[1])), 4, (0, 60, 255), -1, cv2.LINE_AA)

    cv2.polylines(image, [centers.reshape((-1, 1, 2))],
                  isClosed=False, color=(0, 255, 80), thickness=2, lineType=cv2.LINE_AA)

    for left_pt, right_pt in zip(mid_p[0::2, :], mid_p[1::2, :]):
        cv2.line(image,
                 (int(left_pt[0]), int(left_pt[1])),
                 (int(right_pt[0]), int(right_pt[1])),
                 color=(0, 180, 255), thickness=1, lineType=cv2.LINE_AA)

    pair_colors = {
        'Main': (0, 255, 255),
        'Upper': (255, 255, 0),
        'Lower': (255, 80, 180),
    }
    label_x = int(w * 0.68)
    label_y = 45
    drawn_lines = set()

    for label, angle, idx1, idx2 in angle_pairs:
        color = pair_colors.get(label, (0, 255, 0))
        right_points = []
        for idx in [idx1, idx2]:
            if idx < 0 or idx >= len(mid_p) // 2:
                continue
            p1 = mid_p[idx * 2]
            p2 = mid_p[idx * 2 + 1]
            line_key = int(idx)
            line_start, line_end = _clip_extended_line(p1, p2, w, h)
            if line_key not in drawn_lines:
                cv2.line(image, line_start, line_end, color=color, thickness=2, lineType=cv2.LINE_AA)
                drawn_lines.add(line_key)
            right_points.append(max([line_start, line_end], key=lambda point: point[0]))

        if len(right_points) == 2:
            cv2.line(image, right_points[0], right_points[1], color=color, thickness=1, lineType=cv2.LINE_AA)
            anchor = ((right_points[0][0] + right_points[1][0]) / 2 + 8,
                      (right_points[0][1] + right_points[1][1]) / 2)
        else:
            anchor = (label_x, label_y)

        _put_label(image, '{}: {:.1f} deg'.format(label, angle), anchor, color)
        label_y += 34

    return [angle for _, angle, _, _ in angle_pairs]


def cobb_angle_calc(pts, image):
    pts = np.asarray(pts, np.float32)   # 68 x 2
    h,w,c = image.shape
    num_pts = pts.shape[0]   # number of points, 68
    vnum = num_pts//4-1

    mid_p_v = (pts[0::2,:]+pts[1::2,:])/2   # 34 x 2
    mid_p = []
    for i in range(0, num_pts, 4):
        pt1 = (pts[i,:]+pts[i+2,:])/2
        pt2 = (pts[i+1,:]+pts[i+3,:])/2
        mid_p.append(pt1)
        mid_p.append(pt2)
    mid_p = np.asarray(mid_p, np.float32)   # 34 x 2

    for pt in mid_p:
        cv2.circle(image,
                   (int(pt[0]), int(pt[1])),
                   12, (0,255,255), -1, 1)

    for pt1, pt2 in zip(mid_p[0::2,:], mid_p[1::2,:]):
        cv2.line(image,
                 (int(pt1[0]), int(pt1[1])),
                 (int(pt2[0]), int(pt2[1])),
                 color=(0,0,255),
                 thickness=5, lineType=1)

    vec_m = mid_p[1::2,:]-mid_p[0::2,:]           # 17 x 2
    dot_v = np.matmul(vec_m, np.transpose(vec_m)) # 17 x 17
    mod_v = np.sqrt(np.sum(vec_m**2, axis=1))[:, np.newaxis]    # 17 x 1
    mod_v = np.matmul(mod_v, np.transpose(mod_v)) # 17 x 17
    cosine_angles = np.clip(dot_v/mod_v, a_min=0., a_max=1.)
    angles = np.arccos(cosine_angles)   # 17 x 17
    pos1 = np.argmax(angles, axis=1)
    maxt = np.amax(angles, axis=1)
    pos2 = np.argmax(maxt)
    cobb_angle1 = np.amax(maxt)
    cobb_angle1 = cobb_angle1/np.pi*180
    flag_s = is_S(mid_p_v)
    if not flag_s: # not S
        # print('Not S')
        cobb_angle2 = angles[0, pos2]/np.pi*180
        cobb_angle3 = angles[vnum, pos1[pos2]]/np.pi*180
        cv2.line(image,
                 (int(mid_p[pos2 * 2, 0] ), int(mid_p[pos2 * 2, 1])),
                 (int(mid_p[pos2 * 2 + 1, 0]), int(mid_p[pos2 * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)
        cv2.line(image,
                 (int(mid_p[pos1[pos2] * 2, 0]), int(mid_p[pos1[pos2] * 2, 1])),
                 (int(mid_p[pos1[pos2] * 2 + 1, 0]), int(mid_p[pos1[pos2] * 2 + 1, 1])),
                 color=(0, 255, 0), thickness=5, lineType=2)

    else:
        if (mid_p_v[pos2*2, 1]+mid_p_v[pos1[pos2]*2,1])<h:
            # print('Is S: condition1')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1[pos2], pos1[pos2]:(vnum+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180
            pos1_2 = pos1_2 + pos1[pos2]-1

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

        else:
            # print('Is S: condition2')
            angle2 = angles[pos2,:(pos2+1)]
            cobb_angle2 = np.max(angle2)
            pos1_1 = np.argmax(angle2)
            cobb_angle2 = cobb_angle2/np.pi*180

            angle3 = angles[pos1_1, :(pos1_1+1)]
            cobb_angle3 = np.max(angle3)
            pos1_2 = np.argmax(angle3)
            cobb_angle3 = cobb_angle3/np.pi*180

            cv2.line(image,
                     (int(mid_p[pos1_1 * 2, 0]), int(mid_p[pos1_1 * 2, 1])),
                     (int(mid_p[pos1_1 * 2+1, 0]), int(mid_p[pos1_1 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

            cv2.line(image,
                     (int(mid_p[pos1_2 * 2, 0]), int(mid_p[pos1_2 * 2, 1])),
                     (int(mid_p[pos1_2 * 2+1, 0]), int(mid_p[pos1_2 * 2 + 1, 1])),
                     color=(0, 255, 0), thickness=5, lineType=2)

    return [cobb_angle1, cobb_angle2, cobb_angle3]
