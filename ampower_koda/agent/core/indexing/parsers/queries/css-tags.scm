; CSS tags. Shared by .css and .scss.
;
; Thin by nature: a stylesheet has selectors, not symbols. What it does buy is
; the ability to answer "where is `.koda-ide-sidebar` defined?" with a location
; instead of a grep, which is the question anyone actually asks of a stylesheet.
;
; `class_name` is captured rather than its inner `identifier`, because the CSS
; and SCSS grammars nest that differently and the node text is the same either
; way.

(rule_set (selectors (class_selector (class_name) @name))) @definition.class
(rule_set (selectors (id_selector (id_name) @name))) @definition.class

(keyframes_statement (keyframes_name) @name) @definition.constant
