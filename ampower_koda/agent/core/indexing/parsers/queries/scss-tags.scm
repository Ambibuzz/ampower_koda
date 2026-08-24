; SCSS tags.
;
; Everything the CSS query captures, plus the three constructs that make SCSS a
; language rather than a format: mixins, functions, and the `@include` sites
; that use them.
;
; This is a separate file rather than a shared one with a few extra patterns,
; because a pattern naming a node type the CSS grammar has never heard of is not
; an inert pattern — it fails the whole query at compile time. One grammar, one
; query, and the duplication is four lines.

(rule_set (selectors (class_selector (class_name) @name))) @definition.class
(rule_set (selectors (id_selector (id_name) @name))) @definition.class

(keyframes_statement (keyframes_name) @name) @definition.constant

(mixin_statement name: (identifier) @name) @definition.function
(function_statement name: (identifier) @name) @definition.function

(include_statement (identifier) @name) @reference.call
