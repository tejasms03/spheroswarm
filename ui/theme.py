"""The dock's visual language, shared by the apps that use it.

Lifted out of `app.py` unchanged rather than imported from it: `app.py` pulls
in torch, the LLM client and the whole tool layer at module scope, and a
bring-up tool has to start on a machine where any of those is broken. That is
precisely the machine you are on when you need it.
"""

import pygame

INK = (11, 34, 57)
PANEL = (15, 44, 70)
PANEL2 = (20, 55, 84)
RULE = (44, 78, 108)
CHALK = (228, 240, 248)
DIM = (143, 179, 204)
CYAN = (99, 210, 232)
SUN = (245, 179, 66)
CORAL = (255, 107, 107)
MINT = (126, 226, 168)
GREY = (110, 124, 140)
CARD = (14, 40, 65)
CARD_EDGE = (28, 60, 90)

PAD = 14
GAP = 16

LED_RGB = {
    "red": (255, 60, 60), "yellow": (255, 210, 70), "green": (90, 220, 120),
    "cyan": (99, 210, 232), "blue": (90, 140, 250), "magenta": (225, 100, 215),
}


def section(surface, font, label, x, y, w, right_text=None, right_color=None):
    """An uppercase, letter-spaced section header. Returns the next y."""
    surface.blit(font.render(" ".join(label.upper()), True, (120, 156, 186)), (x, y))
    if right_text:
        t = font.render(right_text, True, right_color or (120, 156, 186))
        surface.blit(t, (x + w - t.get_width(), y))
    pygame.draw.line(surface, (26, 56, 84), (x, y + 15), (x + w, y + 15))
    return y + 24


def card(surface, rect, fill=CARD, edge=CARD_EDGE):
    pygame.draw.rect(surface, fill, rect, border_radius=5)
    pygame.draw.rect(surface, edge, rect, 1, border_radius=5)
    return rect


def stat_tile(surface, small, big, rect, label, value, tone=None):
    card(surface, rect)
    surface.blit(small.render(" ".join(label.upper()), True, (108, 142, 172)),
                 (rect.x + 8, rect.y + 6))
    surface.blit(big.render(str(value), True, tone or CHALK), (rect.x + 8, rect.y + 20))


class Button:
    def __init__(self, rect, label, cb, toggle=False, tone=None):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.cb = cb
        self.toggle = toggle
        self.tone = tone
        self.on = False
        self.enabled = True

    def draw(self, s, f):
        col = self.tone or (CYAN if self.on else RULE)
        bg = PANEL2 if self.on else PANEL
        if not self.enabled:
            col, bg = RULE, INK
        pygame.draw.rect(s, bg, self.rect, border_radius=3)
        pygame.draw.rect(s, col, self.rect, 1, border_radius=3)
        t = f.render(self.label, True, CHALK if self.enabled else RULE)
        s.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, p):
        if self.enabled and self.rect.collidepoint(p):
            self.cb()
            return True
        return False


class Slider:
    """A labelled integer slider. Drag the track or click anywhere on it."""

    def __init__(self, rect, label, lo, hi, get, set_):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.lo, self.hi = lo, hi
        self.get, self.set = get, set_
        self.dragging = False

    @property
    def track(self):
        r = self.rect
        return pygame.Rect(r.x + 92, r.y + 6, r.w - 132, 4)

    def draw(self, s, f):
        v = int(self.get())
        s.blit(f.render(self.label, True, DIM), (self.rect.x, self.rect.y))
        t = self.track
        pygame.draw.rect(s, RULE, t, border_radius=2)
        frac = (v - self.lo) / max(self.hi - self.lo, 1)
        x = int(t.x + frac * t.w)
        pygame.draw.rect(s, CYAN, (t.x, t.y, x - t.x, t.h), border_radius=2)
        pygame.draw.circle(s, CYAN, (x, t.y + 2), 6)
        s.blit(f.render(str(v), True, CHALK), (t.right + 10, self.rect.y))

    def hit(self, p):
        t = self.track
        if not pygame.Rect(t.x - 8, self.rect.y - 4, t.w + 16, self.rect.h + 8).collidepoint(p):
            return False
        self.dragging = True
        self.drag(p)
        return True

    def drag(self, p):
        t = self.track
        frac = min(1.0, max(0.0, (p[0] - t.x) / max(t.w, 1)))
        self.set(int(round(self.lo + frac * (self.hi - self.lo))))
