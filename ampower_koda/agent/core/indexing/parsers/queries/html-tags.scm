; HTML tags.
;
; Two jobs, and the second is the important one.
;
; First, elements carrying an `id` become addressable: a Frappe page template is
; mostly a skeleton of mount points, and `#koda-ide-root` is the name the
; JavaScript reaches for.
;
; Second, and this is why the file exists at all: an HTML `<script>` body is a
; single `raw_text` node to this grammar. Without the injection below, every
; symbol in an inline script would be invisible — the file would index as prose.

((element (start_tag
   (attribute
     (attribute_name) @_attribute
     (quoted_attribute_value (attribute_value) @name)) @definition.field))
 (#eq? @_attribute "id"))

(script_element (raw_text) @injection.javascript)
(style_element (raw_text) @injection.css)
